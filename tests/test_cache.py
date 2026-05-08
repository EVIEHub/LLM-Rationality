"""Unit tests for the sample cache.

Covers fingerprint stability, immutability (no overwrite), atomic
write, header validation on read, type coercion, and the
``cache_exists`` query.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from src.pipeline.cache import (
    CACHE_FORMAT_VERSION,
    cache_exists,
    cache_path,
    read_cache,
    write_cache,
)


# ---------------------------------------------------------------------------
# Fingerprint stability and uniqueness
# ---------------------------------------------------------------------------


def test_fingerprint_is_deterministic(cache_key_factory) -> None:
    key1 = cache_key_factory()
    key2 = cache_key_factory()
    assert key1.fingerprint() == key2.fingerprint()


def test_fingerprint_differs_for_each_field(cache_key_factory) -> None:
    """Every field that contributes to sampling must change the fingerprint."""
    base = cache_key_factory()
    perturbations = [
        cache_key_factory(model="other/Model"),
        cache_key_factory(dataset="math"),
        cache_key_factory(K=8),
        cache_key_factory(temperature=0.7),
        cache_key_factory(top_p=0.95),
        cache_key_factory(top_k=50),
        cache_key_factory(max_tokens=2048),
        cache_key_factory(seed=43),
        cache_key_factory(prompt_template_version="v2"),
        cache_key_factory(num_prompts=500),
        cache_key_factory(max_reasoning_length=128),
    ]
    fps = {p.fingerprint() for p in perturbations}
    assert base.fingerprint() not in fps
    assert len(fps) == len(perturbations), "every field-perturbation must give a unique fingerprint"


def test_max_reasoning_length_None_is_distinct_from_zero(cache_key_factory) -> None:
    """v3 invariant: max_reasoning_length=None ("no budget forcing") and
    max_reasoning_length=0 ("budget forcing with 0 reasoning tokens") are
    semantically different and must produce different fingerprints."""
    none_key = cache_key_factory(max_reasoning_length=None)
    zero_key = cache_key_factory(max_reasoning_length=0)
    assert none_key.fingerprint() != zero_key.fingerprint()


def test_int_and_float_canonicalised_for_temperature(cache_key_factory) -> None:
    """temperature=1 and temperature=1.0 must hash identically (post __post_init__)."""
    assert (
        cache_key_factory(temperature=1).fingerprint()
        == cache_key_factory(temperature=1.0).fingerprint()
    )


def test_int_and_float_canonicalised_for_K(cache_key_factory) -> None:
    assert cache_key_factory(K=4).fingerprint() == cache_key_factory(K=4.0).fingerprint()


def test_int_and_float_canonicalised_for_num_prompts(cache_key_factory) -> None:
    """v2: num_prompts is in the key; canonical type is int."""
    assert (
        cache_key_factory(num_prompts=200).fingerprint()
        == cache_key_factory(num_prompts=200.0).fingerprint()  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Path layout
# ---------------------------------------------------------------------------


def test_cache_path_includes_readable_hints(tmp_path: Path, cache_key_factory) -> None:
    key = cache_key_factory(dataset="gsm8k", K=4, model="org/MyModel-7B")
    path = cache_path(tmp_path, key)
    assert path.parent == tmp_path
    assert path.suffix == ".gz"
    assert ".jsonl" in path.name
    assert "gsm8k" in path.name
    assert "MyModel-7B" in path.name
    assert "K4" in path.name
    assert f"v{CACHE_FORMAT_VERSION}" in path.name


def test_cache_path_is_stable_for_same_key(tmp_path: Path, cache_key_factory) -> None:
    key1 = cache_key_factory()
    key2 = cache_key_factory()
    assert cache_path(tmp_path, key1) == cache_path(tmp_path, key2)


def test_cache_path_differs_for_different_keys(tmp_path: Path, cache_key_factory) -> None:
    a = cache_key_factory(seed=1)
    b = cache_key_factory(seed=2)
    assert cache_path(tmp_path, a) != cache_path(tmp_path, b)


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


def test_write_then_read_roundtrip(tmp_path: Path, cache_key_factory) -> None:
    key = cache_key_factory()
    records = [
        {"prompt_id": "p1", "samples": ["a", "b", "c", "d"]},
        {"prompt_id": "p2", "samples": ["e", "f", "g", "h"]},
    ]
    written = write_cache(tmp_path, key, iter(records))
    assert written.exists()
    read_back = list(read_cache(tmp_path, key))
    assert read_back == records


def test_cache_exists_reflects_filesystem(tmp_path: Path, cache_key_factory) -> None:
    key = cache_key_factory()
    assert not cache_exists(tmp_path, key)
    write_cache(tmp_path, key, [{"x": 1}])
    assert cache_exists(tmp_path, key)


# ---------------------------------------------------------------------------
# Immutability (AGENT.md §3.2)
# ---------------------------------------------------------------------------


def test_write_refuses_to_overwrite(tmp_path: Path, cache_key_factory) -> None:
    key = cache_key_factory()
    write_cache(tmp_path, key, [{"x": 1}])
    with pytest.raises(FileExistsError) as excinfo:
        write_cache(tmp_path, key, [{"x": 2}])
    assert "immutable" in str(excinfo.value).lower()


def test_failed_write_leaves_no_partial_file(tmp_path: Path, cache_key_factory) -> None:
    """A crashed write must clean up its .tmp; the target must not exist."""
    key = cache_key_factory()

    def bad_records():
        yield {"x": 1}
        raise RuntimeError("simulated mid-write crash")

    with pytest.raises(RuntimeError, match="simulated"):
        write_cache(tmp_path, key, bad_records())

    target = cache_path(tmp_path, key)
    assert not target.exists(), "target file must not exist after crashed write"
    assert not (target.parent / (target.name + ".tmp")).exists(), ".tmp must be cleaned up"


# ---------------------------------------------------------------------------
# Read validation
# ---------------------------------------------------------------------------


def test_read_missing_cache_raises(tmp_path: Path, cache_key_factory) -> None:
    with pytest.raises(FileNotFoundError):
        list(read_cache(tmp_path, cache_key_factory()))


def test_read_detects_key_mismatch(tmp_path: Path, cache_key_factory) -> None:
    """If a file at the expected path was somehow written under a different
    key (manual rename, hash collision), read must refuse rather than
    silently return mismatched data."""
    key_a = cache_key_factory(seed=1)
    key_b = cache_key_factory(seed=2)

    # Write under key_a, then move the file to where key_b would look for it.
    write_cache(tmp_path, key_a, [{"x": 1}])
    path_a = cache_path(tmp_path, key_a)
    path_b = cache_path(tmp_path, key_b)
    path_a.rename(path_b)

    with pytest.raises(ValueError) as excinfo:
        list(read_cache(tmp_path, key_b))
    assert "mismatch" in str(excinfo.value).lower()


def test_read_detects_format_version_mismatch(tmp_path: Path, cache_key_factory) -> None:
    """A file from an older format version must not be silently consumed."""
    key = cache_key_factory()
    path = cache_path(tmp_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        # Write a header claiming an older format version.
        fh.write(
            json.dumps(
                {
                    "_cache_key": dict(
                        model=key.model,
                        dataset=key.dataset,
                        K=key.K,
                        temperature=key.temperature,
                        top_p=key.top_p,
                        top_k=key.top_k,
                        max_tokens=key.max_tokens,
                        seed=key.seed,
                        prompt_template_version=key.prompt_template_version,
                        num_prompts=key.num_prompts,
                        max_reasoning_length=key.max_reasoning_length,
                    ),
                    "_format_version": CACHE_FORMAT_VERSION - 1,
                }
            )
            + "\n"
        )
        fh.write(json.dumps({"x": 1}) + "\n")

    with pytest.raises(ValueError, match="format version"):
        list(read_cache(tmp_path, key))


def test_read_detects_empty_file(tmp_path: Path, cache_key_factory) -> None:
    key = cache_key_factory()
    path = cache_path(tmp_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as fh:
        pass  # write nothing
    with pytest.raises(ValueError, match="empty"):
        list(read_cache(tmp_path, key))


def test_read_detects_malformed_header(tmp_path: Path, cache_key_factory) -> None:
    key = cache_key_factory()
    path = cache_path(tmp_path, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("this is not json\n")
    with pytest.raises(ValueError, match="malformed header"):
        list(read_cache(tmp_path, key))


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_read_returns_iterator_not_list(tmp_path: Path, cache_key_factory) -> None:
    """Memory-efficient streaming read: the return is an iterator."""
    key = cache_key_factory()
    write_cache(tmp_path, key, [{"x": 1}, {"x": 2}])
    result = read_cache(tmp_path, key)
    assert hasattr(result, "__next__"), "read_cache should return an iterator"


def test_write_accepts_generator_input(tmp_path: Path, cache_key_factory) -> None:
    """Writers can stream records through a generator without materialising them."""
    key = cache_key_factory()

    def gen():
        for i in range(5):
            yield {"prompt_id": f"p{i}", "x": i}

    write_cache(tmp_path, key, gen())
    records = list(read_cache(tmp_path, key))
    assert len(records) == 5
    assert records[0] == {"prompt_id": "p0", "x": 0}
    assert records[4] == {"prompt_id": "p4", "x": 4}
