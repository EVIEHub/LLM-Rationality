"""Structural tests for the YAML configuration files under ``configs/``.

These tests load each YAML and verify the schema invariants documented
in the file headers. They exist so typos and missing keys are caught
at test time rather than mysteriously inside scripts at run time, and
so cross-file references (e.g. an experiment lists a model alias that
exists in models.yaml) are validated together.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.verification.interface import known_datasets

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


@pytest.fixture(scope="module")
def models_yaml() -> dict:
    with open(CONFIGS / "models.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def datasets_yaml() -> dict:
    with open(CONFIGS / "datasets.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def h1_yaml() -> dict:
    with open(CONFIGS / "experiments" / "h1.yaml") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# models.yaml
# ---------------------------------------------------------------------------

REQUIRED_MODEL_KEYS = {"hf_id", "family", "size_b", "prompt_mode", "supports_system_prompt"}


def test_models_yaml_loads(models_yaml) -> None:
    assert isinstance(models_yaml, dict)
    assert len(models_yaml) > 0


def test_models_yaml_contains_smoke_test_model(models_yaml) -> None:
    """The smoke test (Phase 3.10) targets qwen2.5-1.5b-instruct."""
    assert "qwen2.5-1.5b-instruct" in models_yaml


def test_models_yaml_contains_full_tulu_trajectory(models_yaml) -> None:
    """H2 needs base/SFT/DPO/RLVR all present."""
    for alias in ("tulu3-8b-base", "tulu3-8b-sft", "tulu3-8b-dpo", "tulu3-8b-rlvr"):
        assert alias in models_yaml, f"Tülu trajectory missing {alias}"


def test_models_yaml_contains_h1_panel(models_yaml) -> None:
    """H1 cross-model panel: Tulu-3 RLVR + Qwen-2.5-7B + Llama-3.1-8B."""
    for alias in ("tulu3-8b-rlvr", "qwen2.5-7b-instruct", "llama3.1-8b-instruct"):
        assert alias in models_yaml, f"H1 panel missing {alias}"


def test_every_model_has_required_keys(models_yaml) -> None:
    for alias, entry in models_yaml.items():
        missing = REQUIRED_MODEL_KEYS - set(entry)
        assert not missing, f"{alias} missing required keys: {missing}"


def test_prompt_mode_is_chat_or_few_shot(models_yaml) -> None:
    valid = {"chat", "few_shot"}
    for alias, entry in models_yaml.items():
        assert entry["prompt_mode"] in valid, (
            f"{alias} has invalid prompt_mode: {entry['prompt_mode']!r}"
        )


def test_tulu_trajectory_stages_are_correctly_tagged(models_yaml) -> None:
    """All four Tülu trajectory stages must have the right `trajectory_stage`."""
    expected = {
        "tulu3-8b-base": "base",
        "tulu3-8b-sft": "sft",
        "tulu3-8b-dpo": "dpo",
        "tulu3-8b-rlvr": "rlvr",
    }
    for alias, expected_stage in expected.items():
        actual = models_yaml[alias].get("trajectory_stage")
        assert actual == expected_stage, (
            f"{alias} trajectory_stage = {actual!r}, expected {expected_stage!r}"
        )


def test_size_b_is_positive(models_yaml) -> None:
    for alias, entry in models_yaml.items():
        assert entry["size_b"] > 0, f"{alias} has non-positive size_b: {entry['size_b']}"


def test_base_model_uses_few_shot(models_yaml) -> None:
    """Tülu base must be in few-shot mode (no chat template)."""
    assert models_yaml["tulu3-8b-base"]["prompt_mode"] == "few_shot"
    assert models_yaml["tulu3-8b-base"]["supports_system_prompt"] is False


# ---------------------------------------------------------------------------
# datasets.yaml
# ---------------------------------------------------------------------------

REQUIRED_DATASET_KEYS = {
    "hf_id",
    "split",
    "verifier",
    "prompt_field",
    "ground_truth_field",
    "templates",
    "prompt_template_version",
    "test_size",
}


def test_datasets_yaml_loads(datasets_yaml) -> None:
    assert isinstance(datasets_yaml, dict)


def test_datasets_yaml_covers_all_three_datasets(datasets_yaml) -> None:
    assert set(datasets_yaml) >= {"gsm8k", "math", "humaneval"}


def test_every_dataset_has_required_keys(datasets_yaml) -> None:
    for alias, entry in datasets_yaml.items():
        missing = REQUIRED_DATASET_KEYS - set(entry)
        assert not missing, f"{alias} missing required keys: {missing}"


def test_every_dataset_verifier_resolves_in_registry(datasets_yaml) -> None:
    """Each dataset's `verifier` field must match a name registered in
    src/verification/interface.py — otherwise scripts will fail at runtime
    with a confusing KeyError."""
    registered = set(known_datasets())
    for alias, entry in datasets_yaml.items():
        verifier = entry["verifier"]
        assert verifier in registered, (
            f"{alias} references verifier {verifier!r} not in registry {sorted(registered)}"
        )


def test_every_dataset_has_chat_and_few_shot_templates(datasets_yaml) -> None:
    for alias, entry in datasets_yaml.items():
        assert "chat" in entry["templates"], f"{alias} missing chat template"
        assert "few_shot" in entry["templates"], f"{alias} missing few_shot template"
        assert "user_template" in entry["templates"]["chat"]
        assert "user_template" in entry["templates"]["few_shot"]


def test_humaneval_has_entry_point_field(datasets_yaml) -> None:
    """HumanEval verification needs the entry-point function name to
    assemble the check program; the dataset config must record where to
    find it."""
    assert "entry_point_field" in datasets_yaml["humaneval"]


# ---------------------------------------------------------------------------
# experiments/h1.yaml
# ---------------------------------------------------------------------------


def test_h1_yaml_loads(h1_yaml) -> None:
    assert h1_yaml["hypothesis"] == "h1"


def test_h1_panel_has_three_models_three_datasets(h1_yaml) -> None:
    assert len(h1_yaml["models"]) == 3
    assert len(h1_yaml["datasets"]) == 3


def test_h1_K_grid_subset_of_K_max(h1_yaml) -> None:
    """K subsampling estimator requires every K' in the curve to be <= K_max."""
    K_max = h1_yaml["K_max"]
    for K_prime in h1_yaml["K_grid_for_curve"]:
        assert K_prime <= K_max, f"K_grid contains {K_prime} > K_max={K_max}"


def test_h1_K_reference_is_in_grid(h1_yaml) -> None:
    """K=64 reference scalar must be a point on the saturation curve."""
    assert h1_yaml["K_reference"] in h1_yaml["K_grid_for_curve"]


def test_h1_uses_three_seeds(h1_yaml) -> None:
    """AGENT.md §3.5 default: variance reported across 3 seeds."""
    assert len(h1_yaml["seeds"]) == 3


def test_h1_seeds_are_distinct(h1_yaml) -> None:
    assert len(set(h1_yaml["seeds"])) == 3


def test_h1_models_resolve_in_models_yaml(h1_yaml, models_yaml) -> None:
    for alias in h1_yaml["models"]:
        assert alias in models_yaml, f"H1 references unknown model alias: {alias}"


def test_h1_datasets_resolve_in_datasets_yaml(h1_yaml, datasets_yaml) -> None:
    for alias in h1_yaml["datasets"]:
        assert alias in datasets_yaml, f"H1 references unknown dataset alias: {alias}"


def test_h1_sampling_defaults_match_AGENT_section_3_1(h1_yaml) -> None:
    """AGENT.md §3.1: temperature=1.0, top_p=1.0, top_k=-1. H1 must NOT
    deviate — H3 is the experiment that varies sampling."""
    sampling = h1_yaml["sampling"]
    assert sampling["temperature"] == 1.0
    assert sampling["top_p"] == 1.0
    assert sampling["top_k"] == -1


def test_h1_bootstrap_matches_methodology_memo(h1_yaml) -> None:
    """Methodology memo: B=1000 resamples, 95% CI."""
    bootstrap = h1_yaml["bootstrap"]
    assert bootstrap["n_resamples"] == 1000
    assert bootstrap["confidence"] == 0.95


def test_h1_all_models_use_chat_mode(h1_yaml, models_yaml) -> None:
    """H1 panel is intentionally instruct-only; mixing in a base model
    would conflate "no alignment" with "rational gap exists" — H2 is the
    experiment for trajectory effects."""
    for alias in h1_yaml["models"]:
        assert models_yaml[alias]["prompt_mode"] == "chat", (
            f"{alias} is not in chat mode; H1 must be instruct-only"
        )
