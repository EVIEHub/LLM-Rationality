"""API-based sampler — drop-in alternative to :class:`VllmRunner` for
hosted (closed-weight) models. Generation only; no GPU.

Used for H5 API subjects (the H1 saturation and H3 procedure analyses)
on deployment benchmarks. Mirrors :meth:`VllmRunner.sample`'s contract:
``(M prompts, K) -> List[List[str]]`` of shape ``(M, K)``.

Differences from :class:`VllmRunner`, all driven by the realities of a
rate-limited / quota-capped HTTP proxy:

- **Async batched HTTP** with a concurrency semaphore (proxies 429/503
  above a modest in-flight count).
- **Exponential-backoff retry** on 429 / 503 / timeouts / transport
  errors.
- **Per-(prompt, k) resume cache.** Every completed call is appended to
  a JSONL sidecar keyed by ``(i, k)``. A quota cutoff mid-run therefore
  loses nothing — rerun resumes from where it stopped. This is essential
  for ChatGPT-account proxies with daily message caps.
- **Chat messages built internally** from a system prompt + per-prompt
  user text. There is no local tokenizer / chat template — the hosted
  model applies its own.

The OpenAI-style ``n`` parameter (K completions in one request) is
intentionally *not* used: issuing K independent requests gives
finer-grained resume and sidesteps proxies that reject ``n>1``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.sampling.vllm_runner import SamplingConfig

logger = logging.getLogger(__name__)

# Retry policy for transient proxy failures. Includes the Cloudflare-style
# 52x codes and 544 ("Error return from script") seen from the DeepSeek
# proxy — all server-side/transient, so safe to retry rather than abort a cell.
#
# 429 is handled separately by `_is_retryable_429`: the same HTTP code is
# returned for two very different conditions on quota-capped proxies (e.g.
# a third-party ChatGPT-account gateway used by gpt-5.2-chat / gpt-5.5):
#   - "API_KEY_QUOTA_EXHAUSTED" (daily account quota used up): NOT retryable —
#     retrying floods the proxy with rejected requests and trips its
#     anti-abuse, disabling the key (HTTP 401 API_KEY_DISABLED).
#   - "Upstream rate limit exceeded" / "rate_limit_error" (per-minute upstream
#     throttle): retryable — backing off and trying again works.
# The body-substring discriminator below distinguishes the two without
# touching the proxy. For metered APIs (DeepSeek), 429 is rare and falls
# through to the retryable branch by default.
_RETRYABLE_STATUS = {408, 409, 500, 502, 503, 504, 520, 521, 522, 524, 544}
_RETRYABLE_429_BODY_SUBSTRINGS = ("rate_limit", "Upstream", "upstream", "Too Many Requests")
_NON_RETRYABLE_429_BODY_SUBSTRINGS = ("QUOTA_EXHAUSTED", "quota", "exhausted")
_MAX_RETRIES = 6


def _is_retryable_429(body: str) -> bool:
    """Return True for transient upstream rate limits, False for quota-exhausted.

    Defaults to False (non-retryable) when neither substring matches — quota
    issues on quota-capped proxies must never silently retry into a key ban.
    """
    if any(s in body for s in _NON_RETRYABLE_429_BODY_SUBSTRINGS):
        return False
    if any(s in body for s in _RETRYABLE_429_BODY_SUBSTRINGS):
        return True
    return False
_BACKOFF_BASE_S = 2.0
_BACKOFF_MAX_S = 60.0


@dataclass(frozen=True)
class ApiModelSpec:
    """Resolved API endpoint configuration for one model alias."""

    api_model: str           # model name sent in the request body
    base_url: str            # e.g. https://.../v1
    api_key: str             # bearer token
    reasoning_effort: str | None = None  # e.g. "minimal" to disable long reasoning

    @classmethod
    def from_model_cfg(cls, model_cfg: dict[str, Any]) -> "ApiModelSpec":
        """Build from a ``models.yaml`` entry with ``backend: api``.

        Reads the key + base URL from the env vars named in the config
        (never inlines secrets into the config file).
        """
        base_env = model_cfg["api_base_url_env"]
        key_env = model_cfg["api_key_env"]
        base_url = os.environ.get(base_env)
        api_key = os.environ.get(key_env)
        if not base_url:
            raise RuntimeError(
                f"API base URL env var {base_env!r} is unset; "
                f"source the env file before running an API model."
            )
        if not api_key:
            raise RuntimeError(
                f"API key env var {key_env!r} is unset; "
                f"source the env file before running an API model."
            )
        return cls(
            api_model=model_cfg.get("api_model", model_cfg["hf_id"]),
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            reasoning_effort=model_cfg.get("reasoning_effort"),
        )


class ApiRunner:
    """HTTP sampler with the same ``sample`` contract as :class:`VllmRunner`.

    Args:
        spec: resolved endpoint config.
        system: system-prompt string applied to every request (may be "").
        resume_path: JSONL sidecar for per-(prompt, k) resume. If the
            file exists, its entries are loaded and those calls skipped.
        concurrency: max in-flight requests (proxy rate-limit ceiling).
        request_timeout_s: per-request timeout.
    """

    def __init__(
        self,
        spec: ApiModelSpec,
        system: str = "",
        resume_path: str | Path | None = None,
        concurrency: int = 10,
        request_timeout_s: float = 180.0,
    ) -> None:
        self.spec = spec
        self.system = system
        self.resume_path = Path(resume_path) if resume_path else None
        self.concurrency = concurrency
        self.request_timeout_s = request_timeout_s

    # -- resume cache --------------------------------------------------------

    def _load_resume(self) -> dict[tuple[int, int], str]:
        done: dict[tuple[int, int], str] = {}
        if self.resume_path and self.resume_path.exists():
            with self.resume_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        done[(rec["i"], rec["k"])] = rec["text"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a torn final line
        return done

    def _append_resume(self, i: int, k: int, text: str) -> None:
        if not self.resume_path:
            return
        self.resume_path.parent.mkdir(parents=True, exist_ok=True)
        with self.resume_path.open("a") as fh:
            fh.write(json.dumps({"i": i, "k": k, "text": text}) + "\n")

    # -- HTTP ----------------------------------------------------------------

    async def _one_call(
        self,
        client: Any,
        user: str,
        cfg: SamplingConfig,
        seed: int,
    ) -> str:
        """One chat completion with retry. Returns the message content."""
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": user})
        body = {
            "model": self.spec.api_model,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "seed": seed,  # best-effort reproducibility hint
        }
        # Disable long reasoning for reasoning models that would otherwise
        # exceed a proxy gateway timeout (e.g. gpt-5.5 -> 504). Keeps the
        # subject comparable to the other non-thinking chat models.
        if self.spec.reasoning_effort:
            body["reasoning_effort"] = self.spec.reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.spec.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.spec.base_url}/chat/completions"

        last_err = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                r = await client.post(
                    url, json=body, headers=headers,
                    timeout=self.request_timeout_s,
                )
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"] or ""
                last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                if r.status_code == 429:
                    # 429 has two sub-types on quota-capped proxies; the body
                    # tells them apart (see `_is_retryable_429`).
                    if not _is_retryable_429(r.text):
                        raise RuntimeError(f"non-retryable {last_err}")
                elif r.status_code not in _RETRYABLE_STATUS:
                    raise RuntimeError(f"non-retryable {last_err}")
            except Exception as e:  # noqa: BLE001 — retry transport + 5xx alike
                last_err = f"{type(e).__name__}: {e}"
                if "non-retryable" in last_err:
                    raise
            if attempt < _MAX_RETRIES:
                delay = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_MAX_S)
                delay *= 0.5 + random.random()  # jitter
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"API call failed after {_MAX_RETRIES + 1} attempts: {last_err}"
        )

    def _make_client(self) -> Any:
        """Return an async HTTP client context manager.

        Isolated so tests can inject a fake client without httpx or a
        network. Production path uses :class:`httpx.AsyncClient`.
        """
        import httpx

        return httpx.AsyncClient()

    async def _run_async(
        self,
        prompts: list[str],
        K: int,
        seed: int,
        cfg: SamplingConfig,
    ) -> list[list[str]]:
        done = self._load_resume()
        n_cached = len(done)
        results: dict[tuple[int, int], str] = dict(done)
        sem = asyncio.Semaphore(self.concurrency)
        write_lock = asyncio.Lock()

        async with self._make_client() as client:
            async def worker(i: int, k: int) -> None:
                async with sem:
                    text = await self._one_call(
                        client, prompts[i], cfg, seed=seed * 100003 + k,
                    )
                results[(i, k)] = text
                async with write_lock:
                    self._append_resume(i, k, text)

            tasks = [
                worker(i, k)
                for i in range(len(prompts))
                for k in range(K)
                if (i, k) not in done
            ]
            if tasks:
                logger.info(
                    "ApiRunner: %d calls to issue (%d already cached), "
                    "concurrency=%d", len(tasks), n_cached, self.concurrency,
                )
                await asyncio.gather(*tasks)

        return [[results[(i, k)] for k in range(K)] for i in range(len(prompts))]

    # -- public contract (mirrors VllmRunner.sample) -------------------------

    def sample(
        self,
        prompts: list[str],
        K: int,
        seed: int,
        config: SamplingConfig | None = None,
    ) -> list[list[str]]:
        """Sample ``K`` completions per prompt via the HTTP API.

        Args:
            prompts: ``M`` *user-message* strings (raw question text — the
                hosted model applies its own chat template; do NOT
                pre-format with a local tokenizer).
            K: per-prompt sample budget; ``>= 1``.
            seed: RNG seed (best-effort; diversity across K relies on
                ``temperature > 0``).
            config: sampling parameters; AGENT.md §3.1 defaults if None.

        Returns:
            ``M`` lists of ``K`` strings each, shape ``(M, K)``.
        """
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        cfg = config or SamplingConfig()
        if cfg.temperature == 0.0 and K > 1:
            raise ValueError(
                "Greedy decoding (temperature=0.0) with K>1 is wasteful: "
                "all K samples will be identical. Use K=1 for greedy."
            )
        return asyncio.run(self._run_async(prompts, K, seed, cfg))

    def close(self) -> None:
        """No-op; present for interface parity with :class:`VllmRunner`."""

    def __enter__(self) -> "ApiRunner":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
