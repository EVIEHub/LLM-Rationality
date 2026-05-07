"""Path configuration for the rational-gap pipeline.

All runtime data — sample cache, results, raw datasets, logs — lives
**outside** the repository. This module is the single source of truth
for those locations: scripts call :func:`load_paths` rather than
hardcoding paths or assembling them from environment variables (per
AGENT.md §3.4).

The configuration file is ``configs/paths.yaml`` (gitignored). Users
copy ``configs/paths.template.yaml`` and edit ``outputs_root``; the
remaining four entries are typically interpolated from it via
``${outputs_root}``.

Interpolation rules:
- ``${name}`` is substituted with the value of ``name`` in the same
  YAML file. Substitution iterates to a fixed point so chains of
  references resolve.
- ``~`` is expanded to the current user's home directory.
- Cyclic references raise :class:`ValueError`.
- Undefined ``${name}`` references raise :class:`KeyError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATHS_YAML = _REPO_ROOT / "configs" / "paths.yaml"
_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")
_REQUIRED_KEYS = (
    "outputs_root",
    "samples_dir",
    "results_dir",
    "raw_data_dir",
    "logs_dir",
)


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem locations for all runtime data.

    All attributes are absolute :class:`pathlib.Path` instances. The
    directories are *not* created automatically; call
    :meth:`ensure_dirs` before writing.
    """

    outputs_root: Path
    samples_dir: Path
    results_dir: Path
    raw_data_dir: Path
    logs_dir: Path

    def ensure_dirs(self) -> None:
        """Create all five directories if they do not already exist."""
        for attr in _REQUIRED_KEYS:
            getattr(self, attr).mkdir(parents=True, exist_ok=True)


def _resolve_string(
    value: str,
    raw: dict[str, Any],
    seen: frozenset[str] = frozenset(),
) -> str:
    """Substitute ``${name}`` references in ``value`` until a fixed point.

    Args:
        value: The string to resolve.
        raw: The full mapping of names to raw (possibly unresolved) values.
        seen: Names currently being expanded — used to detect cycles.

    Returns:
        The string with all ``${name}`` references replaced.

    Raises:
        KeyError: A referenced name is not in ``raw``.
        ValueError: Expansion would cycle.
    """
    previous = None
    while previous != value:
        previous = value

        def _sub(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in seen:
                raise ValueError(
                    f"Cyclic interpolation detected at ${{{name}}} (chain: {sorted(seen)})"
                )
            if name not in raw:
                known = ", ".join(sorted(raw))
                raise KeyError(f"Undefined path variable ${{{name}}} (known: {known})")
            child = raw[name]
            if not isinstance(child, str):
                return str(child)
            return _resolve_string(child, raw, seen | {name})

        value = _VAR_PATTERN.sub(_sub, value)
    return value


def load_paths(path: Path | str | None = None) -> Paths:
    """Load and resolve the paths configuration.

    Args:
        path: Optional explicit path to the YAML config. Defaults to
            ``<repo>/configs/paths.yaml``. Pass an explicit path in
            tests to load a fixture without touching the user's local
            config.

    Returns:
        A frozen :class:`Paths` with absolute, ``~``-expanded paths.

    Raises:
        FileNotFoundError: The config file does not exist (likely the
            user has not yet copied ``paths.template.yaml``).
        KeyError: A required key is missing or an interpolation
            references an undefined name.
        ValueError: The YAML root is not a mapping, or interpolation
            would cycle.
    """
    config_path = Path(path) if path is not None else _DEFAULT_PATHS_YAML
    if not config_path.exists():
        raise FileNotFoundError(
            f"Paths config not found at {config_path}. "
            f"Copy configs/paths.template.yaml to configs/paths.yaml "
            f"and edit `outputs_root`."
        )

    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected a YAML mapping at the root of {config_path}, got {type(raw).__name__}"
        )

    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(
            f"Missing required keys in {config_path}: {missing}. "
            f"Required: {list(_REQUIRED_KEYS)}"
        )

    resolved = {k: _resolve_string(v, raw) if isinstance(v, str) else v for k, v in raw.items()}

    return Paths(
        outputs_root=Path(resolved["outputs_root"]).expanduser(),
        samples_dir=Path(resolved["samples_dir"]).expanduser(),
        results_dir=Path(resolved["results_dir"]).expanduser(),
        raw_data_dir=Path(resolved["raw_data_dir"]).expanduser(),
        logs_dir=Path(resolved["logs_dir"]).expanduser(),
    )
