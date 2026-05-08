"""Alternative inference procedures (H3, H4) layered on top of VllmRunner.

Each procedure has the signature
    procedure(runner, prompts, K, seed, **kwargs) -> list[list[str]]
matching :meth:`VllmRunner.sample` so the H1/H2 sampling-and-aggregate
pipeline can swap them in without further changes.

Currently implemented:
    budget_forced(...)       — H4: s1-style two-stage budget forcing.

Not implemented (skipped per HANDOFF.md plan C):
    cot, self_consistency, mcts, direct_t0, direct_t07
"""

from __future__ import annotations

from typing import Any

from src.sampling.vllm_runner import SamplingConfig, VllmRunner


def budget_forced(
    runner: VllmRunner,
    prompts: list[str],
    K: int,
    seed: int,
    *,
    max_reasoning_length: int,
    answer_prompt: str = "\n\nFinal answer:",
    answer_max_tokens: int = 64,
) -> list[list[str]]:
    """Two-stage budget-forced sampling (s1-paper-style).

    Stage 1: sample K completions per prompt with ``max_tokens=max(1, L)``.
        This caps the model's reasoning at ``L`` tokens. If the model
        finishes earlier (e.g. emits an EOS) the stage 1 output is
        shorter; if it would have continued, it is truncated.
    Stage 2: append ``answer_prompt`` to every (prompt, stage1_completion)
        pair and sample one more completion of up to ``answer_max_tokens``
        tokens per pair. This forces the model to commit an answer even
        if its reasoning was cut short.

    The returned strings are the full ``stage1 + answer_prompt + stage2``
    text per (prompt, k), so downstream verifiers see the same surface
    as H1/H2 (extract the final number / boxed expression / code).

    Args:
        runner: VllmRunner with the model already loaded.
        prompts: $M$ prompt strings.
        K: per-prompt sample budget.
        seed: base seed; stage 2 uses ``seed + 1`` so its RNG state is
            independent of stage 1's K-fold sampling.
        max_reasoning_length: $L$ — the reasoning token budget. ``L=0``
            is allowed and means "no reasoning" (vLLM still requires
            ``max_tokens >= 1`` so we use 1 internally; the resulting
            stage-1 token contributes negligibly to the answer).
        answer_prompt: text appended after stage 1 to elicit the final
            answer. Default ``"\\n\\nFinal answer:"`` is generic; for
            datasets with specific answer markers (``####``, ``\\boxed{}``)
            the verifier handles either format.
        answer_max_tokens: ``max_tokens`` for stage 2. ``64`` is plenty
            for a numeric answer or a short LaTeX expression.

    Returns:
        ``(M, K)`` list-of-lists of strings.
    """
    if max_reasoning_length < 0:
        raise ValueError(
            f"max_reasoning_length must be >= 0, got {max_reasoning_length}"
        )

    # Stage 1: bounded reasoning.
    L = max(1, max_reasoning_length)
    cfg1 = SamplingConfig(
        temperature=1.0, top_p=1.0, top_k=-1, max_tokens=L,
    )
    stage1 = runner.sample(prompts, K=K, seed=seed, config=cfg1)

    # Stage 2: build M*K independent prompts (each a unique
    # original + stage1_completion + answer_prompt) and sample one
    # continuation per. K=1 is correct: each prompt is unique, so we
    # only need one continuation each.
    M = len(prompts)
    stage2_prompts: list[str] = []
    for i in range(M):
        for k in range(K):
            stage2_prompts.append(prompts[i] + stage1[i][k] + answer_prompt)

    cfg2 = SamplingConfig(
        temperature=1.0, top_p=1.0, top_k=-1, max_tokens=answer_max_tokens,
    )
    stage2_flat = runner.sample(
        stage2_prompts, K=1, seed=seed + 1, config=cfg2,
    )  # shape (M*K, 1)

    output: list[list[str]] = []
    idx = 0
    for i in range(M):
        cell: list[str] = []
        for k in range(K):
            full_text = stage1[i][k] + answer_prompt + stage2_flat[idx][0]
            cell.append(full_text)
            idx += 1
        output.append(cell)
    return output
