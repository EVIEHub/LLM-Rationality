"""LLM-as-judge verifier for the preference-evaluation setting.

The verifier compares a candidate response $y_k$ from the model under
test against a human-preferred reference $y^+$ for the same prompt $x$,
and returns a ternary outcome:

    o_k =  1.0  if  y_k  >  y^+   (win)
    o_k =  0.5  if  y_k  ~  y^+   (tie or ambiguous)
    o_k =  0.0  if  y_k  <  y^+   (lose)

Each (prompt, candidate, reference) triple is judged $L$ times by an
LLM acting as judge (sampled at non-zero temperature for i.i.d.
verdicts). Position is randomised per call to control position bias.
Final $o_k$ is assigned by strict majority over the $L$ votes; if no
class clears the majority threshold, $o_k = 0.5$.

This verifier is **not** a pure function — it needs an
:class:`src.sampling.vllm_runner.VllmRunner` instance to make judge
calls. Cell runners (``scripts/run_h{1,2}.py`` in preference mode)
hand it the runner explicitly.

The interface is **batched** for efficiency: a single call to
:func:`score_matrix` makes all $M \\times K \\times L$ judge requests
through vLLM in one batched pass, then aggregates back into an
$(M, K)$ utility matrix that downstream metrics code consumes
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.sampling.vllm_runner import SamplingConfig, VllmRunner


# Strict-majority threshold for L=5 votes: ceil(L/2) = 3
# (and more generally: ceil(L/2) for any L).
_JUDGE_TEMPERATURE = 0.7
_JUDGE_MAX_TOKENS = 8


@dataclass(frozen=True)
class JudgeOutcome:
    """Per-cell judge bookkeeping. Useful for audit / debugging."""

    utility: np.ndarray            # (M, K) float in {0, 0.5, 1}
    raw_verdicts: np.ndarray       # (M, K, L) float in {0, 0.5, 1} — per-call outcomes
    parse_failure_rate: float      # fraction of judge outputs that did not parse to A/B/T
    n_judge_calls: int             # total calls = M * K * L


def _build_judge_prompt(
    tokenizer,
    user_question: str,
    response_a: str,
    response_b: str,
) -> str:
    """Build a chat-formatted judge prompt with a deterministic schema.

    The judge is instructed to emit a single letter (A / B / T) so we
    can parse with the first non-whitespace character of the response.
    """
    system = (
        "You are evaluating two responses to a user's question. "
        "Pick the better one. Answer with exactly one character: "
        "A if Response A is better, B if Response B is better, "
        "T if they are equally good."
    )
    user = (
        f"User question:\n{user_question}\n\n"
        f"--- Response A ---\n{response_a}\n\n"
        f"--- Response B ---\n{response_b}\n\n"
        f"Which is better? Answer with one letter (A, B, or T):"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _parse_verdict(text: str, a_is_candidate: bool) -> Optional[float]:
    """Parse the judge's output and convert to a candidate-relative score.

    Args:
        text: Raw judge output (first 8 tokens of generation).
        a_is_candidate: True if response A was the candidate $y_k$, False if
            A was the reference $y^+$.

    Returns:
        ``1.0`` if the judge picked the candidate, ``0.0`` if it picked the
        reference, ``0.5`` for tie. ``None`` if the output did not parse to
        A/B/T — caller should treat ``None`` as ``0.5`` (ambiguous).
    """
    s = text.strip()
    if not s:
        return None
    c = s[0].upper()
    if c == "A":
        return 1.0 if a_is_candidate else 0.0
    if c == "B":
        return 0.0 if a_is_candidate else 1.0
    if c == "T":
        return 0.5
    return None


def _aggregate(votes: list[float], majority_threshold: int) -> float:
    """Strict-majority ternary aggregation.

    Args:
        votes: List of $L$ per-call outcomes, each in $\\{0, 0.5, 1\\}$.
            ``None`` (parse failure) is mapped to 0.5 by the caller before
            this function sees them.
        majority_threshold: Minimum same-class votes for a verdict;
            ``ceil(L/2)``. With L=5 this is 3.

    Returns:
        ``1.0`` if win-class has majority,
        ``0.0`` if lose-class has majority,
        ``0.5`` if neither (tie majority or no consensus).
    """
    n_win = sum(1 for v in votes if v == 1.0)
    n_lose = sum(1 for v in votes if v == 0.0)
    if n_win >= majority_threshold:
        return 1.0
    if n_lose >= majority_threshold:
        return 0.0
    return 0.5


def score_matrix(
    judge_runner: VllmRunner,
    judge_tokenizer,
    raw_prompts: list[str],
    candidates: list[list[str]],
    references: list[str],
    *,
    L: int = 5,
    seed: int = 0,
) -> JudgeOutcome:
    """Batched LLM-as-judge scoring for a full $(M, K)$ candidate matrix.

    For each of the $M \\times K$ (prompt, candidate, reference) triples,
    we make $L$ i.i.d. judge calls with **random position per call**
    (A=candidate or A=reference, decided by a per-call coin flip seeded
    deterministically from ``(seed, i, k, l)``). All $M \\times K \\times L$
    judge prompts are issued in one batched vLLM call.

    Args:
        judge_runner: A live :class:`VllmRunner`. May or may not be the
            same model that produced the candidates (H1 uses the same
            model; H2 uses a fixed Tulu-3-RLVR judge across stages).
        judge_tokenizer: The judge model's tokenizer (used to apply its
            chat template).
        raw_prompts: ``M`` raw user questions (not chat-formatted).
        candidates: ``M`` lists of ``K`` candidate response strings.
        references: ``M`` reference response strings ($y^+$).
        L: Judge calls per (prompt, candidate) pair. Default 5.
        seed: Seed for both the position-randomisation RNG and vLLM
            sampling.

    Returns:
        :class:`JudgeOutcome` with the aggregated $(M, K)$ utility
        matrix plus per-call raw verdicts and parse-failure rate for
        auditability.

    Raises:
        ValueError: On shape mismatch between candidates and references.
    """
    M = len(raw_prompts)
    if len(candidates) != M or len(references) != M:
        raise ValueError(
            f"shape mismatch: prompts={M}, candidates={len(candidates)}, "
            f"references={len(references)}"
        )
    K = len(candidates[0])
    for i, c in enumerate(candidates):
        if len(c) != K:
            raise ValueError(f"prompt {i}: K={len(c)} differs from prompt 0's K={K}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    majority_threshold = (L + 1) // 2  # ceil(L/2)

    rng = np.random.default_rng(seed)
    # Position assignment: shape (M, K, L) bool. True = candidate is in
    # position A. Done up front so we can re-derive verdicts from raw
    # judge outputs without re-flipping coins.
    a_is_candidate = rng.random(size=(M, K, L)) < 0.5

    # Build the M * K * L judge prompts in a flat list.
    judge_prompts: list[str] = []
    for i in range(M):
        for k in range(K):
            for l in range(L):
                if a_is_candidate[i, k, l]:
                    a, b = candidates[i][k], references[i]
                else:
                    a, b = references[i], candidates[i][k]
                judge_prompts.append(
                    _build_judge_prompt(judge_tokenizer, raw_prompts[i], a, b)
                )

    # Batched judge call. K=1 here means "one sampled output per prompt"
    # (vLLM's K param); we are NOT re-using the L axis inside vLLM.
    cfg = SamplingConfig(
        temperature=_JUDGE_TEMPERATURE,
        top_p=1.0, top_k=-1,
        max_tokens=_JUDGE_MAX_TOKENS,
    )
    flat_outputs = judge_runner.sample(
        judge_prompts, K=1, seed=seed, config=cfg,
    )
    assert len(flat_outputs) == M * K * L, (
        f"vllm returned {len(flat_outputs)} outputs, expected {M * K * L}"
    )

    # Parse + aggregate.
    raw_verdicts = np.full((M, K, L), 0.5, dtype=np.float32)
    utility = np.zeros((M, K), dtype=np.float32)
    n_parse_fail = 0
    idx = 0
    for i in range(M):
        for k in range(K):
            votes = []
            for l in range(L):
                output_text = flat_outputs[idx][0]  # K=1 → single sample
                parsed = _parse_verdict(output_text, bool(a_is_candidate[i, k, l]))
                if parsed is None:
                    n_parse_fail += 1
                    vote = 0.5
                else:
                    vote = parsed
                raw_verdicts[i, k, l] = vote
                votes.append(vote)
                idx += 1
            utility[i, k] = _aggregate(votes, majority_threshold)

    return JudgeOutcome(
        utility=utility,
        raw_verdicts=raw_verdicts,
        parse_failure_rate=n_parse_fail / (M * K * L) if M * K * L > 0 else 0.0,
        n_judge_calls=M * K * L,
    )
