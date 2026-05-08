"""Unit tests for inference procedures (H3, H4 alternative samplers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.sampling.inference_procedures import budget_forced
from src.sampling.vllm_runner import VllmRunner


def _build_mock_request(texts: list[str]) -> MagicMock:
    req = MagicMock()
    req.outputs = [MagicMock(text=t) for t in texts]
    return req


@pytest.fixture
def mock_llm():
    with patch("vllm.LLM") as MockLLM:
        instance = MockLLM.return_value
        yield instance


def test_budget_forced_calls_generate_twice(mock_llm) -> None:
    """Two-stage: stage 1 (M prompts × K) + stage 2 (M*K prompts × 1)."""
    M, K = 2, 3
    # Stage 1 returns K completions per prompt
    stage1_returns = [
        _build_mock_request([f"reason_p{i}_k{k}" for k in range(K)])
        for i in range(M)
    ]
    # Stage 2 returns 1 completion per (M*K) flattened prompt
    stage2_returns = [_build_mock_request([f"answer_{j}"]) for j in range(M * K)]

    mock_llm.generate.side_effect = [stage1_returns, stage2_returns]

    runner = VllmRunner("fake/model")
    out = budget_forced(
        runner,
        ["p0", "p1"],
        K=K,
        seed=42,
        max_reasoning_length=128,
    )

    # generate was called exactly twice (one per stage)
    assert mock_llm.generate.call_count == 2

    # Output shape: (M, K)
    assert len(out) == M
    for cell in out:
        assert len(cell) == K

    # Each output is stage1 + answer_prompt + stage2
    for i in range(M):
        for k in range(K):
            assert "reason_" in out[i][k]
            assert "Final answer:" in out[i][k]
            assert "answer_" in out[i][k]


def test_budget_forced_stage1_max_tokens_matches_L(mock_llm) -> None:
    """The stage-1 SamplingParams must have max_tokens == max(1, L)."""
    mock_llm.generate.side_effect = [
        [_build_mock_request(["s1"])],  # stage 1
        [_build_mock_request(["s2"])],  # stage 2
    ]
    runner = VllmRunner("fake/model")
    budget_forced(runner, ["p"], K=1, seed=0, max_reasoning_length=256)

    stage1_params = mock_llm.generate.call_args_list[0][0][1]
    assert stage1_params.max_tokens == 256


def test_budget_forced_stage2_uses_K_equals_one(mock_llm) -> None:
    """Stage 2 has unique prompt per (i, k), so n=K=1 is correct."""
    M, K = 3, 2
    mock_llm.generate.side_effect = [
        [_build_mock_request([f"s{k}" for k in range(K)]) for _ in range(M)],
        [_build_mock_request(["a"]) for _ in range(M * K)],
    ]
    runner = VllmRunner("fake/model")
    budget_forced(runner, ["p0", "p1", "p2"], K=K, seed=0, max_reasoning_length=128)

    stage2_params = mock_llm.generate.call_args_list[1][0][1]
    assert stage2_params.n == 1


def test_budget_forced_L_zero_is_allowed(mock_llm) -> None:
    """L=0 means no reasoning; vLLM still gets max_tokens>=1 internally."""
    mock_llm.generate.side_effect = [
        [_build_mock_request(["x"])],
        [_build_mock_request(["a"])],
    ]
    runner = VllmRunner("fake/model")
    budget_forced(runner, ["p"], K=1, seed=0, max_reasoning_length=0)

    stage1_params = mock_llm.generate.call_args_list[0][0][1]
    assert stage1_params.max_tokens == 1  # clamped to >= 1


def test_budget_forced_L_negative_raises(mock_llm) -> None:
    runner = VllmRunner("fake/model")
    with pytest.raises(ValueError, match="must be >= 0"):
        budget_forced(runner, ["p"], K=1, seed=0, max_reasoning_length=-1)


def test_budget_forced_uses_separate_stage_seeds(mock_llm) -> None:
    """Stage 1 uses `seed`, stage 2 uses `seed + 1` (independent RNGs)."""
    mock_llm.generate.side_effect = [
        [_build_mock_request(["s"])],
        [_build_mock_request(["a"])],
    ]
    runner = VllmRunner("fake/model")
    budget_forced(runner, ["p"], K=1, seed=42, max_reasoning_length=64)

    s1_seed = mock_llm.generate.call_args_list[0][0][1].seed
    s2_seed = mock_llm.generate.call_args_list[1][0][1].seed
    assert s1_seed == 42
    assert s2_seed == 43
