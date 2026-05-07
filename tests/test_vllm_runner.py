"""Unit + live integration tests for the vLLM runner.

The file is split into:

- mock-based tests that exercise the wrapper's parameter-packing and
  refusal logic without loading a real model;
- live tests on a tiny model (Qwen-2.5-0.5B-Instruct, ~1 GB) that pin
  the byte-determinism contract from AGENT.md §3.1. Per
  ``methodology/no_skipif_for_invariants.md`` these are NOT gated
  behind ``@pytest.mark.skipif(no_gpu)``.

The live tests are kept tiny: a 4-prompt fixture, ``max_tokens=32``,
and a single shared :class:`VllmRunner` for the whole module via
session-scoped fixture, so the suite finishes in seconds once the
0.5B weights are cached.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.sampling.vllm_runner import SamplingConfig, VllmRunner


# ---------------------------------------------------------------------------
# Mock-based tests — fast, no GPU touch
# ---------------------------------------------------------------------------


def _build_mock_request(texts: list[str]) -> MagicMock:
    """Build a fake vLLM RequestOutput with ``len(texts)`` completions."""
    req = MagicMock()
    req.outputs = [MagicMock(text=t) for t in texts]
    return req


@pytest.fixture
def mock_llm():
    """Patch ``vllm.LLM`` so VllmRunner never instantiates a real engine."""
    with patch("vllm.LLM") as MockLLM:
        instance = MockLLM.return_value
        # By default, .generate returns one RequestOutput with K texts.
        # Tests that need a different shape override `instance.generate`.
        yield instance


def test_sample_calls_generate_once_with_n_equals_K(mock_llm) -> None:
    """AGENT.md §3.1 / §8.4: the wrapper must use vLLM's n=K, NOT loop K
    times. Verify ``generate`` is called exactly once and the
    SamplingParams it receives has ``n == K``."""
    mock_llm.generate.return_value = [_build_mock_request([f"s{i}" for i in range(8)])]
    runner = VllmRunner("fake/model")
    runner.sample(["prompt"], K=8, seed=42)

    assert mock_llm.generate.call_count == 1, "generate must be called exactly once"
    params = mock_llm.generate.call_args[0][1]
    assert params.n == 8


def test_sample_default_params_match_AGENT_3_1(mock_llm) -> None:
    """AGENT.md §3.1: defaults are temperature=1.0, top_p=1.0,
    top_k=-1, max_tokens (we use 2048 here as the SamplingConfig
    default). Seed is whatever the caller passed."""
    mock_llm.generate.return_value = [_build_mock_request(["s"])]
    runner = VllmRunner("fake/model")
    runner.sample(["prompt"], K=1, seed=99)

    params = mock_llm.generate.call_args[0][1]
    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == -1
    assert params.max_tokens == 2048
    assert params.seed == 99


def test_sample_returns_M_by_K_strings(mock_llm) -> None:
    """Shape contract: outputs[i][k] is the k-th completion for prompt i."""
    mock_llm.generate.return_value = [
        _build_mock_request(["a", "b", "c"]),
        _build_mock_request(["d", "e", "f"]),
    ]
    runner = VllmRunner("fake/model")
    out = runner.sample(["p1", "p2"], K=3, seed=0)

    assert out == [["a", "b", "c"], ["d", "e", "f"]]


def test_sample_K_below_one_raises(mock_llm) -> None:
    runner = VllmRunner("fake/model")
    with pytest.raises(ValueError, match="K must be >= 1"):
        runner.sample(["prompt"], K=0, seed=0)


def test_sample_greedy_with_K_above_one_raises(mock_llm) -> None:
    """AGENT.md §8.1: greedy + K>1 is wasteful; refuse explicitly so the
    caller knows to use K=1 for greedy."""
    runner = VllmRunner("fake/model")
    cfg = SamplingConfig(temperature=0.0)
    with pytest.raises(ValueError, match="Greedy"):
        runner.sample(["prompt"], K=4, seed=0, config=cfg)


def test_sample_greedy_with_K_one_is_allowed(mock_llm) -> None:
    """K=1 + temperature=0 is the legitimate greedy path."""
    mock_llm.generate.return_value = [_build_mock_request(["greedy"])]
    runner = VllmRunner("fake/model")
    cfg = SamplingConfig(temperature=0.0)
    out = runner.sample(["prompt"], K=1, seed=0, config=cfg)

    assert out == [["greedy"]]


def test_close_makes_subsequent_sample_raise(mock_llm) -> None:
    """After close, the runner refuses further sampling rather than
    silently re-instantiating."""
    runner = VllmRunner("fake/model")
    # close() imports torch which we patch to a no-op, so the test
    # doesn't require CUDA.
    with patch("torch.cuda.empty_cache"):
        runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.sample(["prompt"], K=1, seed=0)


def test_close_is_idempotent(mock_llm) -> None:
    runner = VllmRunner("fake/model")
    with patch("torch.cuda.empty_cache"):
        runner.close()
        runner.close()  # second call must not raise


def test_sample_passes_custom_config(mock_llm) -> None:
    """Custom SamplingConfig values flow through to SamplingParams."""
    mock_llm.generate.return_value = [_build_mock_request(["s"])]
    runner = VllmRunner("fake/model")
    cfg = SamplingConfig(temperature=0.7, top_p=0.95, top_k=50, max_tokens=128)
    runner.sample(["prompt"], K=1, seed=1, config=cfg)

    params = mock_llm.generate.call_args[0][1]
    assert params.temperature == 0.7
    assert params.top_p == 0.95
    assert params.top_k == 50
    assert params.max_tokens == 128


# ---------------------------------------------------------------------------
# Live tests — load a real tiny model. Run on the GPU box.
#
# NEVER gate these behind `@pytest.mark.skipif(not torch.cuda.is_available())`
# per methodology/no_skipif_for_invariants.md: byte-determinism is a
# load-bearing methodological invariant from AGENT.md §3.1 and must be
# verified live.
# ---------------------------------------------------------------------------

_TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_TINY_PROMPTS = [
    "What is 2+2?",
    "Name a colour.",
    "Say hello.",
    "What is 5*5?",
]


@pytest.fixture(scope="module")
def tiny_runner():
    """Live VllmRunner with the smallest Qwen-2.5 instruct (~1 GB).

    Module-scoped so the model is loaded once per test session, not per
    test. ``gpu_memory_utilization=0.3`` leaves room on shared GPUs.
    ``enforce_eager=True`` disables CUDA graph capture, which makes
    sampling slightly slower but more deterministic across calls.
    """
    runner = VllmRunner(
        _TINY_MODEL,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    yield runner
    runner.close()


def test_live_shape_matches_M_by_K(tiny_runner) -> None:
    K = 2
    out = tiny_runner.sample(_TINY_PROMPTS, K=K, seed=0)

    assert len(out) == len(_TINY_PROMPTS)
    for completions in out:
        assert len(completions) == K
        for text in completions:
            assert isinstance(text, str)


def test_live_byte_determinism_with_fixed_seed(tiny_runner) -> None:
    """AGENT.md §3.1: same model + prompts + seed must give byte-identical
    samples across two calls. This is the load-bearing determinism
    invariant; per methodology/no_skipif_for_invariants.md it is NEVER
    skipped."""
    cfg = SamplingConfig(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=32)
    out1 = tiny_runner.sample(_TINY_PROMPTS, K=4, seed=42, config=cfg)
    out2 = tiny_runner.sample(_TINY_PROMPTS, K=4, seed=42, config=cfg)

    assert out1 == out2, (
        "vLLM produced different samples for the same (model, prompts, "
        "seed). Either the runner is leaking state across calls, or this "
        "vLLM build does not preserve seed determinism — pin a different "
        "version per AGENT/HANDOFF.md §3.8.3."
    )


def test_live_different_seeds_give_different_samples(tiny_runner) -> None:
    """Sanity check that the seed actually drives variation: two
    different seeds should give different samples on at least some
    prompts (with temperature=1.0, ~probability ≈1 of disagreement)."""
    cfg = SamplingConfig(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=32)
    out_a = tiny_runner.sample(_TINY_PROMPTS, K=2, seed=1, config=cfg)
    out_b = tiny_runner.sample(_TINY_PROMPTS, K=2, seed=2, config=cfg)

    assert out_a != out_b, "different seeds produced identical outputs"
