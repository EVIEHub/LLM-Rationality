"""Sample cache: gzipped JSONL keyed by sampling configuration.

Every sampling configuration that affects model output gets exactly one
cache file. Per AGENT.md §3.2, the cache key includes
``(model, dataset, K, temperature, top_p, top_k, max_tokens, seed,
prompt_template_version)``: missing any of these creates silent bugs
where a cache hit returns samples from a different configuration.

The file format is gzipped JSON lines:

- The first line is a header recording the full cache key and the
  format version.
- Each subsequent line is one prompt's worth of samples, with the
  schema decided by the sampling layer. The cache module is agnostic
  to record contents; it just reads and writes ``dict`` objects.

Cache files are immutable (AGENT.md §3.2). :func:`write_cache` refuses
to overwrite, and writes via a ``.tmp`` sibling + atomic rename so a
crashed run never leaves a half-written file looking like a cache hit.

If the on-disk format changes (not the contents — the format itself,
e.g. switching to Parquet), bump :data:`CACHE_FORMAT_VERSION`. The
version is encoded in the filename, so old and new caches coexist
without silent invalidation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


CACHE_FORMAT_VERSION: int = 2


# Optional fields whose default value is ``None``. When a field's value is
# ``None``, we EXCLUDE it from the fingerprint and the on-disk header.
# This makes new optional fields backward-compatible: an older cache file
# that pre-dates the field still matches a new key whose value of that
# field is ``None``. When the field is set to a non-None value (e.g. H4
# uses ``max_reasoning_length=128``), the fingerprint changes and a new
# cache file is written.
_OPTIONAL_NONE_FIELDS = ("max_reasoning_length",)


@dataclass(frozen=True)
class CacheKey:
    """All parameters that affect sampling output.

    Per AGENT.md §3.2, missing any of these from the cache key creates
    silent bugs where a cache hit returns samples from a different
    configuration.

    Numeric fields are coerced to canonical types in ``__post_init__``,
    so ``CacheKey(K=4, temperature=1)`` and
    ``CacheKey(K=4, temperature=1.0)`` produce identical fingerprints.

    History:
        v2 (2026-05-08): added ``num_prompts`` because subsampling
            (e.g. MATH 5000 -> 1000 for compute budget) means two runs
            with different prompt counts must NOT collide on a single
            cache file. ``CACHE_FORMAT_VERSION`` bumped 1->2 so old
            v1 files (without num_prompts in their header) are not
            silently consumed by the new key shape.
        v2.1 (2026-05-08): added optional ``max_reasoning_length`` for
            H4 budget-forced sampling. ``None`` means "no budget
            forcing" (H1/H2 default); integer values drive two-stage
            budget forcing. CACHE_FORMAT_VERSION did NOT bump because
            optional-with-default-None fields are backward-compatible:
            a v2 file written before this change has the same
            fingerprint as a key whose ``max_reasoning_length=None``.
            See ``_OPTIONAL_NONE_FIELDS``.
    """

    model: str
    dataset: str
    K: int
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    seed: int
    prompt_template_version: str
    num_prompts: int
    max_reasoning_length: int | None = None

    def __post_init__(self) -> None:
        # Frozen-dataclass-safe canonicalisation of numeric types.
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))
        object.__setattr__(self, "top_k", int(self.top_k))
        object.__setattr__(self, "max_tokens", int(self.max_tokens))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "num_prompts", int(self.num_prompts))
        if self.max_reasoning_length is not None:
            object.__setattr__(self, "max_reasoning_length", int(self.max_reasoning_length))

    def canonical_dict(self) -> dict[str, Any]:
        """Return the dict used for fingerprinting and on-disk header.

        Drops ``_OPTIONAL_NONE_FIELDS`` whose value is ``None``, so a key
        that uses default values for every optional field is fingerprint-
        equivalent to a key from a CacheKey schema that did not yet have
        the optional fields. Required for backward compatibility across
        schema additions of optional fields.
        """
        d = asdict(self)
        for name in _OPTIONAL_NONE_FIELDS:
            if d.get(name) is None:
                d.pop(name, None)
        return d

    def fingerprint(self) -> str:
        """Stable 16-hex-char SHA-256 digest of the canonical key."""
        canonical = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def cache_path(samples_dir: Path, key: CacheKey) -> Path:
    """Canonical path for a cache key's data file.

    Filename includes readable hints (dataset, model, K) plus the
    fingerprint hash — the hints help debugging via ``ls`` without
    affecting uniqueness, which the hash guarantees. The format version
    prefix lets old and new formats coexist when
    :data:`CACHE_FORMAT_VERSION` is bumped.
    """
    fp = key.fingerprint()
    model_short = key.model.split("/")[-1].replace(".", "_")
    name = f"v{CACHE_FORMAT_VERSION}_{key.dataset}_{model_short}_K{key.K}_{fp}.jsonl.gz"
    return samples_dir / name


def cache_exists(samples_dir: Path, key: CacheKey) -> bool:
    """``True`` iff the cache file for ``key`` exists on disk."""
    return cache_path(samples_dir, key).exists()


def write_cache(
    samples_dir: Path,
    key: CacheKey,
    records: Iterable[dict[str, Any]],
) -> Path:
    """Write ``records`` to the cache file for ``key``.

    The first line written is a header capturing the full cache key
    and format version, so a corrupted filename can never cause a
    silently-mismatched read.

    Writes go to a ``.tmp`` sibling first and then atomically rename to
    the target — a crashed run leaves either nothing or a complete
    file, never a partial one (AGENT.md §3.2: "cache hit must be
    byte-deterministic").

    Args:
        samples_dir: Directory under which cache files live (typically
            ``Paths.samples_dir``). Created if missing.
        key: The cache key.
        records: An iterable of ``dict`` records to serialise as JSONL.

    Returns:
        The path of the written cache file.

    Raises:
        FileExistsError: A cache file for this key already exists.
            Cache files are immutable; callers must delete explicitly
            to recompute.
    """
    path = cache_path(samples_dir, key)
    if path.exists():
        raise FileExistsError(
            f"Cache for fingerprint {key.fingerprint()} already exists at {path}. "
            f"Cache files are immutable per AGENT.md §3.2; delete the file "
            f"explicitly to recompute."
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.parent / (path.name + ".tmp")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            # Write the canonical dict (matching the fingerprint) so a
            # later CacheKey schema with new optional fields can still
            # validate the header without false key-mismatches.
            header = {
                "_cache_key": key.canonical_dict(),
                "_format_version": CACHE_FORMAT_VERSION,
            }
            fh.write(json.dumps(header, sort_keys=True) + "\n")
            for record in records:
                fh.write(json.dumps(record) + "\n")
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    return path


def read_cache(samples_dir: Path, key: CacheKey) -> Iterator[dict[str, Any]]:
    """Iterate records from the cache file for ``key``.

    Validates that the embedded header matches ``key`` — defends
    against the (vanishingly unlikely) hash collision and against
    files that have been moved or renamed by hand.

    Yields only data records; the internal header line is consumed and
    checked but not yielded.

    Raises:
        FileNotFoundError: No cache file exists for this key.
        ValueError: The file is empty, the header is malformed, the
            embedded key does not match ``key``, or the format version
            does not match :data:`CACHE_FORMAT_VERSION`.
    """
    path = cache_path(samples_dir, key)
    if not path.exists():
        raise FileNotFoundError(
            f"No cache file for fingerprint {key.fingerprint()} at {path}"
        )

    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header_line = fh.readline()
        if not header_line:
            raise ValueError(f"Cache file {path} is empty (missing header)")
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cache file {path} has malformed header: {exc}") from exc

        on_disk_version = header.get("_format_version")
        if on_disk_version != CACHE_FORMAT_VERSION:
            raise ValueError(
                f"Cache file {path} has format version {on_disk_version}, "
                f"current is {CACHE_FORMAT_VERSION}"
            )
        on_disk_key = header.get("_cache_key")
        expected = key.canonical_dict()
        if on_disk_key != expected:
            raise ValueError(
                f"Cache file {path} key mismatch: file recorded {on_disk_key}, "
                f"caller asked for {expected}"
            )

        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
