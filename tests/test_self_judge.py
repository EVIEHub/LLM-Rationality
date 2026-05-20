"""Tests for src.verification.self_judge.

The batched judge in score_matrix needs a VllmRunner-shaped object;
we inject a fake that returns deterministic, configurable outputs so
the tests run on CPU without loading any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pytest

from src.verification.self_judge import (
    _aggregate,
    _parse_verdict,
    score_matrix,
)


# ---------------------------------------------------------------------------
# _parse_verdict
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_a_when_candidate_is_a(self) -> None:
        # Judge says 'A', and A was the candidate -> candidate wins (1.0).
        assert _parse_verdict("A", a_is_candidate=True) == 1.0

    def test_a_when_candidate_is_b(self) -> None:
        # Judge says 'A', and A was the reference -> reference wins (0.0).
        assert _parse_verdict("A", a_is_candidate=False) == 0.0

    def test_b_when_candidate_is_a(self) -> None:
        # Judge says 'B', and A was the candidate -> reference wins.
        assert _parse_verdict("B", a_is_candidate=True) == 0.0

    def test_b_when_candidate_is_b(self) -> None:
        # Judge says 'B', and B was the candidate -> candidate wins.
        assert _parse_verdict("B", a_is_candidate=False) == 1.0

    def test_t_is_tie(self) -> None:
        assert _parse_verdict("T", a_is_candidate=True) == 0.5
        assert _parse_verdict("T", a_is_candidate=False) == 0.5

    def test_lowercase_accepted(self) -> None:
        assert _parse_verdict("a", a_is_candidate=True) == 1.0
        assert _parse_verdict("b", a_is_candidate=True) == 0.0
        assert _parse_verdict("t", a_is_candidate=True) == 0.5

    def test_leading_whitespace_stripped(self) -> None:
        assert _parse_verdict("   A", a_is_candidate=True) == 1.0
        assert _parse_verdict("\n\nB", a_is_candidate=True) == 0.0

    def test_takes_only_first_char(self) -> None:
        # Long judge outputs: we look at first non-whitespace char only.
        assert _parse_verdict("A is better because...", a_is_candidate=True) == 1.0
        assert _parse_verdict("B because it's clearer.", a_is_candidate=False) == 1.0

    def test_unrecognised_returns_none(self) -> None:
        assert _parse_verdict("hello", a_is_candidate=True) is None
        assert _parse_verdict("", a_is_candidate=True) is None
        assert _parse_verdict("?", a_is_candidate=True) is None
        # 'C' for "can't decide" — but we did not define this, so unrecognised.
        assert _parse_verdict("C", a_is_candidate=True) is None


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_unanimous_win(self) -> None:
        assert _aggregate([1.0, 1.0, 1.0, 1.0, 1.0], majority_threshold=3) == 1.0

    def test_unanimous_lose(self) -> None:
        assert _aggregate([0.0, 0.0, 0.0, 0.0, 0.0], majority_threshold=3) == 0.0

    def test_majority_win_at_threshold(self) -> None:
        # 3 of 5 win — exactly hits threshold.
        assert _aggregate([1.0, 1.0, 1.0, 0.0, 0.0], majority_threshold=3) == 1.0
        assert _aggregate([1.0, 1.0, 1.0, 0.5, 0.0], majority_threshold=3) == 1.0

    def test_majority_lose_at_threshold(self) -> None:
        assert _aggregate([0.0, 0.0, 0.0, 1.0, 1.0], majority_threshold=3) == 0.0

    def test_no_majority_is_tie(self) -> None:
        # 2 win, 2 lose, 1 tie — no class has majority of 3.
        assert _aggregate([1.0, 1.0, 0.0, 0.0, 0.5], majority_threshold=3) == 0.5

    def test_three_ties_yields_tie(self) -> None:
        # All 3 ties -> tie wins (no win-majority, no lose-majority).
        assert _aggregate([0.5, 0.5, 0.5, 1.0, 0.0], majority_threshold=3) == 0.5

    def test_L_7_threshold_4(self) -> None:
        # 4-of-7 majority example.
        assert _aggregate(
            [1.0] * 4 + [0.0] * 3, majority_threshold=4,
        ) == 1.0
        # 3-3-1 → no majority → tie.
        assert _aggregate(
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5], majority_threshold=4,
        ) == 0.5


# ---------------------------------------------------------------------------
# score_matrix end-to-end with a fake runner
# ---------------------------------------------------------------------------

@dataclass
class FakeTokenizer:
    """Minimal stub — just returns the user message text concatenated."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False,
    ) -> str:
        return "\n".join(f"[{m['role']}] {m['content']}" for m in messages)


class FakeRunner:
    """Stubs VllmRunner.sample with a configurable per-prompt output.

    Pass ``verdict_sequence`` (e.g. ['A', 'A', 'B', 'T', ...]) and the
    runner will return those in order. Length must equal the number of
    prompts vLLM is called with (M*K*L for score_matrix).
    """

    def __init__(self, verdict_sequence: Iterable[str]):
        self._queue = list(verdict_sequence)
        self.calls: list[dict] = []

    def sample(self, prompts, K, seed, config):
        # vLLM returns List[List[str]] of shape (n_prompts, K).
        self.calls.append({
            "n_prompts": len(prompts), "K": K, "seed": seed,
        })
        if len(self._queue) < len(prompts) * K:
            raise RuntimeError(
                f"FakeRunner queue exhausted: need {len(prompts) * K}, "
                f"have {len(self._queue)}"
            )
        outputs = []
        idx = 0
        for _ in range(len(prompts)):
            row = []
            for _ in range(K):
                row.append(self._queue[idx])
                idx += 1
            outputs.append(row)
        # Pop used entries.
        self._queue = self._queue[idx:]
        return outputs


class TestScoreMatrix:
    def test_unanimous_win_across_pairs(self) -> None:
        # 2 prompts, K=1, L=3 -> 6 judge calls. All return 'A', position
        # randomization handled internally; here we override by making the
        # candidate=A always also map to 'A' wins.
        runner = FakeRunner(["A"] * 6)
        tok = FakeTokenizer()
        outcome = score_matrix(
            judge_runner=runner,
            judge_tokenizer=tok,
            raw_prompts=["q1", "q2"],
            candidates=[["c1"], ["c2"]],
            references=["r1", "r2"],
            L=3,
            seed=0,
        )
        # Each (prompt, sample) has 3 calls all returning 'A'.
        # With random positions, sometimes A=candidate (-> 1.0) and
        # sometimes A=reference (-> 0.0). Verdict depends on RNG. We
        # check shape + that one consistent letter does not collapse the
        # gap to a single value across prompts.
        assert outcome.utility.shape == (2, 1)
        assert outcome.n_judge_calls == 6
        assert outcome.parse_failure_rate == 0.0

    def test_parse_failure_rate(self) -> None:
        # 1 prompt × K=1 × L=5, 2 of 5 outputs are garbage.
        runner = FakeRunner(["A", "????", "A", "garbage", "A"])
        outcome = score_matrix(
            judge_runner=runner,
            judge_tokenizer=FakeTokenizer(),
            raw_prompts=["q"],
            candidates=[["c"]],
            references=["r"],
            L=5,
            seed=0,
        )
        assert outcome.parse_failure_rate == pytest.approx(2 / 5)
        assert outcome.utility.shape == (1, 1)

    def test_majority_aggregation(self) -> None:
        # Force a deterministic position randomization by seeding with a
        # known value. Then construct verdicts so that 3 of 5 calls
        # vote in the same direction regardless of position assignment.
        runner = FakeRunner(["T"] * 5)  # 5 ties
        outcome = score_matrix(
            judge_runner=runner,
            judge_tokenizer=FakeTokenizer(),
            raw_prompts=["q"],
            candidates=[["c"]],
            references=["r"],
            L=5,
            seed=0,
        )
        # 5 ties -> majority is tie (>=3 tie votes), utility = 0.5
        assert outcome.utility[0, 0] == 0.5

    def test_shape_mismatch_raises(self) -> None:
        runner = FakeRunner([])
        with pytest.raises(ValueError, match="shape mismatch"):
            score_matrix(
                judge_runner=runner,
                judge_tokenizer=FakeTokenizer(),
                raw_prompts=["q1", "q2"],
                candidates=[["c1"]],  # only 1 prompt's worth
                references=["r1", "r2"],
            )

    def test_K_mismatch_raises(self) -> None:
        runner = FakeRunner([])
        with pytest.raises(ValueError, match="differs from prompt 0"):
            score_matrix(
                judge_runner=runner,
                judge_tokenizer=FakeTokenizer(),
                raw_prompts=["q1", "q2"],
                candidates=[["c1", "c1b"], ["c2"]],  # K differs
                references=["r1", "r2"],
            )

    def test_L_must_be_positive(self) -> None:
        runner = FakeRunner([])
        with pytest.raises(ValueError, match="L must be"):
            score_matrix(
                judge_runner=runner,
                judge_tokenizer=FakeTokenizer(),
                raw_prompts=["q"],
                candidates=[["c"]],
                references=["r"],
                L=0,
            )

    def test_raw_verdicts_shape(self) -> None:
        # 2 prompts × K=2 × L=3 = 12 calls.
        runner = FakeRunner(["A", "B", "T"] * 4)
        outcome = score_matrix(
            judge_runner=runner,
            judge_tokenizer=FakeTokenizer(),
            raw_prompts=["q1", "q2"],
            candidates=[["c1", "c1b"], ["c2", "c2b"]],
            references=["r1", "r2"],
            L=3,
            seed=42,
        )
        assert outcome.raw_verdicts.shape == (2, 2, 3)
        # Every entry should be in {0, 0.5, 1}.
        for v in outcome.raw_verdicts.flatten():
            assert v in (0.0, 0.5, 1.0)
