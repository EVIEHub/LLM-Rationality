"""API-based LLM-as-judge for the preference setting.

Same contract and aggregation as :mod:`src.verification.self_judge`
(ternary win/tie/lose, L i.i.d. calls per (prompt, candidate) pair,
random A/B position, strict-majority aggregation into an (M, K) utility
matrix) — but the judge is a **hosted API model** (e.g. deepseek-v4-flash)
instead of a local vLLM runner. No GPU: the model under test generates
the candidates on the GPU, then this module judges them over HTTP.

Reuses ``_parse_verdict``, ``_aggregate`` and ``JudgeOutcome`` from
:mod:`self_judge` so the two judge backends produce identical-shaped,
directly-comparable outputs. HTTP plumbing (async batching, retry, the
per-call resume cache) mirrors :class:`src.sampling.api_runner.ApiRunner`,
so a quota cutoff mid-judge loses nothing.

Note on reasoning models (deepseek-v4-flash): we run in default mode and
give ``max_tokens`` enough headroom (256) for the brief chain-of-thought
plus the one-letter verdict, which lands in ``message.content``. The cost
is input-dominated, so the reasoning tokens are a rounding error.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.sampling.api_runner import (
    _BACKOFF_BASE_S,
    _BACKOFF_MAX_S,
    _MAX_RETRIES,
    _RETRYABLE_STATUS,
    ApiModelSpec,
)
from src.verification.self_judge import (
    JudgeOutcome,
    _aggregate,
    _parse_verdict,
)

_JUDGE_SYSTEM = (
    "You are evaluating two responses to a user's question. "
    "Pick the better one. Answer with exactly one character: "
    "A if Response A is better, B if Response B is better, "
    "T if they are equally good."
)
_JUDGE_MAX_TOKENS = 256  # headroom for reasoning models' brief CoT + verdict


def _judge_user(user_question: str, response_a: str, response_b: str) -> str:
    return (
        f"User question:\n{user_question}\n\n"
        f"--- Response A ---\n{response_a}\n\n"
        f"--- Response B ---\n{response_b}\n\n"
        f"Which is better? Answer with one letter (A, B, or T):"
    )


async def _one_judge_call(
    client: Any,
    spec: ApiModelSpec,
    user: str,
    seed: int,
    max_tokens: int,
    request_timeout_s: float,
) -> str:
    """One A/B judge completion with retry. Returns message content."""
    body = {
        "model": spec.api_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,  # i.i.d. verdicts across the L calls
        "seed": seed,
    }
    headers = {
        "Authorization": f"Bearer {spec.api_key}",
        "Content-Type": "application/json",
    }
    url = f"{spec.base_url}/chat/completions"
    last_err = ""
    for attempt in range(_MAX_RETRIES + 1):
        try:
            r = await client.post(url, json=body, headers=headers,
                                  timeout=request_timeout_s)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
            last_err = f"HTTP {r.status_code}: {r.text[:160]}"
            if r.status_code not in _RETRYABLE_STATUS:
                raise RuntimeError(f"non-retryable {last_err}")
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if "non-retryable" in last_err:
                raise
        if attempt < _MAX_RETRIES:
            delay = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_MAX_S)
            delay *= 0.5 + random.random()
            await asyncio.sleep(delay)
    raise RuntimeError(f"judge call failed after {_MAX_RETRIES + 1} tries: {last_err}")


def _make_client() -> Any:
    """Async HTTP client context manager (isolated for test injection)."""
    import httpx

    return httpx.AsyncClient()


async def _run_async(
    spec: ApiModelSpec,
    raw_prompts: list[str],
    candidates: list[list[str]],
    references: list[str],
    a_is_candidate: np.ndarray,
    L: int,
    seed: int,
    concurrency: int,
    max_tokens: int,
    resume_path: Path | None,
    request_timeout_s: float,
    client_factory: Any,
) -> dict[tuple[int, int, int], str]:
    M, K = len(raw_prompts), len(candidates[0])

    done: dict[tuple[int, int, int], str] = {}
    if resume_path and resume_path.exists():
        with resume_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[(rec["i"], rec["k"], rec["l"])] = rec["text"]
                except (json.JSONDecodeError, KeyError):
                    continue

    results: dict[tuple[int, int, int], str] = dict(done)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async with client_factory() as client:
        async def worker(i: int, k: int, l: int) -> None:
            if a_is_candidate[i, k, l]:
                a, b = candidates[i][k], references[i]
            else:
                a, b = references[i], candidates[i][k]
            user = _judge_user(raw_prompts[i], a, b)
            async with sem:
                text = await _one_judge_call(
                    client, spec, user,
                    seed=seed * 100003 + (i * K + k) * L + l,
                    max_tokens=max_tokens, request_timeout_s=request_timeout_s,
                )
            results[(i, k, l)] = text
            if resume_path:
                async with write_lock:
                    resume_path.parent.mkdir(parents=True, exist_ok=True)
                    with resume_path.open("a") as fh:
                        fh.write(json.dumps({"i": i, "k": k, "l": l, "text": text}) + "\n")

        tasks = [
            worker(i, k, l)
            for i in range(M) for k in range(K) for l in range(L)
            if (i, k, l) not in done
        ]
        if tasks:
            await asyncio.gather(*tasks)
    return results


def score_matrix_api(
    spec: ApiModelSpec,
    raw_prompts: list[str],
    candidates: list[list[str]],
    references: list[str],
    *,
    L: int = 3,
    seed: int = 0,
    concurrency: int = 20,
    max_tokens: int = _JUDGE_MAX_TOKENS,
    resume_path: str | Path | None = None,
    request_timeout_s: float = 180.0,
    client_factory: Any = None,
) -> JudgeOutcome:
    """API-judge analogue of :func:`self_judge.score_matrix`.

    Args:
        spec: resolved API endpoint (e.g. deepseek-v4-flash via DS_*).
        raw_prompts: ``M`` raw user questions.
        candidates: ``M`` lists of ``K`` candidate responses.
        references: ``M`` reference responses ($y^+$).
        L: judge calls per (prompt, candidate) pair (default 3 for API).
        seed: position-randomisation + per-call seed.
        concurrency: max in-flight API requests.
        max_tokens: per judge call (256 default; headroom for reasoning
            models' brief CoT before the one-letter verdict).
        resume_path: JSONL sidecar for per-(i,k,l) resume.
        client_factory: async-client factory (tests inject a fake;
            production uses httpx).

    Returns:
        :class:`JudgeOutcome` — identical shape/semantics to the vLLM
        self-judge, so downstream metrics are unchanged.
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
    majority_threshold = (L + 1) // 2

    rng = np.random.default_rng(seed)
    a_is_candidate = rng.random(size=(M, K, L)) < 0.5

    results = asyncio.run(_run_async(
        spec, raw_prompts, candidates, references, a_is_candidate,
        L=L, seed=seed, concurrency=concurrency, max_tokens=max_tokens,
        resume_path=Path(resume_path) if resume_path else None,
        request_timeout_s=request_timeout_s,
        client_factory=client_factory or _make_client,
    ))

    raw_verdicts = np.full((M, K, L), 0.5, dtype=np.float32)
    utility = np.zeros((M, K), dtype=np.float32)
    n_parse_fail = 0
    for i in range(M):
        for k in range(K):
            votes = []
            for l in range(L):
                parsed = _parse_verdict(results[(i, k, l)], bool(a_is_candidate[i, k, l]))
                if parsed is None:
                    n_parse_fail += 1
                    vote = 0.5
                else:
                    vote = parsed
                raw_verdicts[i, k, l] = vote
                votes.append(vote)
            utility[i, k] = _aggregate(votes, majority_threshold)

    n = M * K * L
    return JudgeOutcome(
        utility=utility,
        raw_verdicts=raw_verdicts,
        parse_failure_rate=n_parse_fail / n if n > 0 else 0.0,
        n_judge_calls=n,
    )
