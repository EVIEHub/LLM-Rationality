"""Unit tests for the paths configuration loader.

Covers ``${name}`` interpolation, ``~`` expansion, missing-file and
missing-key error reporting, cycle detection, and the
``ensure_dirs`` helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.paths import Paths, load_paths


def _write_yaml(path: Path, contents: str) -> Path:
    path.write_text(contents)
    return path


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_loads_explicit_paths(tmp_path: Path) -> None:
    config = _write_yaml(
        tmp_path / "paths.yaml",
        f"""
outputs_root: "{tmp_path}/out"
samples_dir: "{tmp_path}/out/data/samples"
results_dir: "{tmp_path}/out/results"
raw_data_dir: "{tmp_path}/out/data/raw"
logs_dir: "{tmp_path}/out/logs"
""",
    )
    paths = load_paths(config)
    assert paths.outputs_root == tmp_path / "out"
    assert paths.samples_dir == tmp_path / "out" / "data" / "samples"
    assert paths.logs_dir == tmp_path / "out" / "logs"


def test_interpolation_resolves_outputs_root(tmp_path: Path) -> None:
    config = _write_yaml(
        tmp_path / "paths.yaml",
        f"""
outputs_root: "{tmp_path}/out"
samples_dir: "${{outputs_root}}/data/samples"
results_dir: "${{outputs_root}}/results"
raw_data_dir: "${{outputs_root}}/data/raw"
logs_dir: "${{outputs_root}}/logs"
""",
    )
    paths = load_paths(config)
    assert paths.samples_dir == tmp_path / "out" / "data" / "samples"
    assert paths.results_dir == tmp_path / "out" / "results"


def test_interpolation_chains(tmp_path: Path) -> None:
    """A path that interpolates a path that interpolates outputs_root."""
    config = _write_yaml(
        tmp_path / "paths.yaml",
        f"""
outputs_root: "{tmp_path}/out"
samples_dir: "${{outputs_root}}/data/samples"
results_dir: "${{samples_dir}}/../../results"
raw_data_dir: "${{outputs_root}}/data/raw"
logs_dir: "${{outputs_root}}/logs"
""",
    )
    paths = load_paths(config)
    assert "samples" in str(paths.results_dir) or paths.results_dir.name == "results"
    # Specifically: samples_dir/../../results = out/results (after Path.resolve)
    assert paths.results_dir == tmp_path / "out" / "data" / "samples" / ".." / ".." / "results"


def test_tilde_is_expanded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _write_yaml(
        tmp_path / "paths.yaml",
        """
outputs_root: "~/rg_out"
samples_dir: "${outputs_root}/samples"
results_dir: "${outputs_root}/results"
raw_data_dir: "${outputs_root}/raw"
logs_dir: "${outputs_root}/logs"
""",
    )
    paths = load_paths(config)
    assert paths.outputs_root == tmp_path / "rg_out"
    assert paths.samples_dir == tmp_path / "rg_out" / "samples"


def test_loads_template_file_unchanged() -> None:
    """The shipped template must be syntactically valid YAML and parse
    without error (after ~ expansion). This catches accidental breakage
    of the template that downstream users will copy."""
    repo_root = Path(__file__).resolve().parents[1]
    template = repo_root / "configs" / "paths.template.yaml"
    paths = load_paths(template)
    # The template uses ~/rational_gap_outputs by default.
    assert paths.outputs_root.is_absolute()
    assert "rational_gap_outputs" in str(paths.outputs_root)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_file_error_mentions_template(tmp_path: Path) -> None:
    nonexistent = tmp_path / "absent.yaml"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_paths(nonexistent)
    msg = str(excinfo.value)
    assert "paths.template.yaml" in msg
    assert "outputs_root" in msg


def test_missing_required_key_raises(tmp_path: Path) -> None:
    config = _write_yaml(
        tmp_path / "paths.yaml",
        """
outputs_root: "/tmp/out"
samples_dir: "/tmp/out/samples"
""",
    )
    with pytest.raises(KeyError) as excinfo:
        load_paths(config)
    msg = str(excinfo.value)
    assert "results_dir" in msg
    assert "raw_data_dir" in msg
    assert "logs_dir" in msg


def test_undefined_interpolation_raises_key_error(tmp_path: Path) -> None:
    config = _write_yaml(
        tmp_path / "paths.yaml",
        """
outputs_root: "/tmp/out"
samples_dir: "${not_defined_anywhere}/samples"
results_dir: "/tmp/out/results"
raw_data_dir: "/tmp/out/raw"
logs_dir: "/tmp/out/logs"
""",
    )
    with pytest.raises(KeyError) as excinfo:
        load_paths(config)
    assert "not_defined_anywhere" in str(excinfo.value)


def test_cyclic_interpolation_raises_value_error(tmp_path: Path) -> None:
    config = _write_yaml(
        tmp_path / "paths.yaml",
        """
outputs_root: "${samples_dir}"
samples_dir: "${outputs_root}/x"
results_dir: "/tmp/results"
raw_data_dir: "/tmp/raw"
logs_dir: "/tmp/logs"
""",
    )
    with pytest.raises(ValueError) as excinfo:
        load_paths(config)
    assert "yclic" in str(excinfo.value)  # "Cyclic" or "cyclic"


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    config = _write_yaml(tmp_path / "paths.yaml", "- a\n- b\n")  # YAML list, not mapping
    with pytest.raises(ValueError) as excinfo:
        load_paths(config)
    assert "mapping" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Paths dataclass behaviour
# ---------------------------------------------------------------------------


def test_paths_is_frozen(tmp_path: Path) -> None:
    paths = Paths(
        outputs_root=tmp_path,
        samples_dir=tmp_path / "s",
        results_dir=tmp_path / "r",
        raw_data_dir=tmp_path / "d",
        logs_dir=tmp_path / "l",
    )
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        paths.outputs_root = tmp_path / "other"  # type: ignore[misc]


def test_ensure_dirs_creates_all(tmp_path: Path) -> None:
    paths = Paths(
        outputs_root=tmp_path / "out",
        samples_dir=tmp_path / "out" / "samples",
        results_dir=tmp_path / "out" / "results",
        raw_data_dir=tmp_path / "out" / "raw",
        logs_dir=tmp_path / "out" / "logs",
    )
    paths.ensure_dirs()
    assert paths.outputs_root.is_dir()
    assert paths.samples_dir.is_dir()
    assert paths.results_dir.is_dir()
    assert paths.raw_data_dir.is_dir()
    assert paths.logs_dir.is_dir()


def test_ensure_dirs_is_idempotent(tmp_path: Path) -> None:
    paths = Paths(
        outputs_root=tmp_path / "out",
        samples_dir=tmp_path / "out" / "samples",
        results_dir=tmp_path / "out" / "results",
        raw_data_dir=tmp_path / "out" / "raw",
        logs_dir=tmp_path / "out" / "logs",
    )
    paths.ensure_dirs()
    paths.ensure_dirs()  # second call must not raise
    assert paths.samples_dir.is_dir()
