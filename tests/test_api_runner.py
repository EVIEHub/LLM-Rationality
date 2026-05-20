"""Tests for src.sampling.api_runner.

No network and no httpx required: a fake async client is injected via
``ApiRunner._make_client`` so the (M, K) assembly, resume-cache, retry,
and config resolution paths run on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sampling.api_runner import ApiModelSpec, ApiRunner
from src.sampling.vllm_runner import SamplingConfig


# ---------------------------------------------------------------------------
# Fake async HTTP client (mimics the subset of httpx we use)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, content: str = "", text: str = ""):
        self.status_code = status_code
        self._content = content
        self.text = text

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
    """Returns a scripted response per call; records request bodies.

    ``responder`` is called with the request json and the running call
    index, and returns a ``_FakeResponse``.
    """

    def __init__(self, responder):
        self._responder = responder
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        idx = len(self.calls)
        self.calls.append(json)
        return self._responder(json, idx)


def _runner_with(responder, **kw) -> ApiRunner:
    spec = ApiModelSpec(api_model="m", base_url="http://x/v1", api_key="k")
    r = ApiRunner(spec, **kw)
    r._make_client = lambda: _FakeClient(responder)  # type: ignore[method-assign]
    return r


# ---------------------------------------------------------------------------
# ApiModelSpec.from_model_cfg
# ---------------------------------------------------------------------------

class TestApiModelSpec:
    def test_resolves_env(self, monkeypatch) -> None:
        monkeypatch.setenv("MY_BASE", "https://proxy/v1/")
        monkeypatch.setenv("MY_KEY", "secret123")
        cfg = {
            "hf_id": "gpt-x", "api_model": "gpt-x",
            "api_base_url_env": "MY_BASE", "api_key_env": "MY_KEY",
        }
        spec = ApiModelSpec.from_model_cfg(cfg)
        assert spec.base_url == "https://proxy/v1"  # trailing slash stripped
        assert spec.api_key == "secret123"
        assert spec.api_model == "gpt-x"

    def test_falls_back_to_hf_id_for_model_name(self, monkeypatch) -> None:
        monkeypatch.setenv("B", "https://p")
        monkeypatch.setenv("K", "k")
        spec = ApiModelSpec.from_model_cfg(
            {"hf_id": "the-model", "api_base_url_env": "B", "api_key_env": "K"}
        )
        assert spec.api_model == "the-model"

    def test_missing_base_url_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("NOPE_BASE", raising=False)
        monkeypatch.setenv("K", "k")
        with pytest.raises(RuntimeError, match="base URL"):
            ApiModelSpec.from_model_cfg(
                {"hf_id": "m", "api_base_url_env": "NOPE_BASE", "api_key_env": "K"}
            )

    def test_missing_key_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("B", "https://p")
        monkeypatch.delenv("NOPE_KEY", raising=False)
        with pytest.raises(RuntimeError, match="key"):
            ApiModelSpec.from_model_cfg(
                {"hf_id": "m", "api_base_url_env": "B", "api_key_env": "NOPE_KEY"}
            )


# ---------------------------------------------------------------------------
# sample(): (M, K) assembly
# ---------------------------------------------------------------------------

class TestSampleShape:
    def test_returns_m_by_k(self) -> None:
        # Echo the call index so we can verify ordering.
        runner = _runner_with(lambda body, idx: _FakeResponse(200, content=f"r{idx}"))
        out = runner.sample(["q0", "q1", "q2"], K=4, seed=0)
        assert len(out) == 3
        assert all(len(row) == 4 for row in out)

    def test_system_prompt_included(self) -> None:
        seen = {}

        def responder(body, idx):
            seen["messages"] = body["messages"]
            return _FakeResponse(200, content="x")

        runner = _runner_with(responder, system="be terse")
        runner.sample(["hello"], K=1, seed=0)
        roles = [m["role"] for m in seen["messages"]]
        assert roles == ["system", "user"]
        assert seen["messages"][0]["content"] == "be terse"
        assert seen["messages"][1]["content"] == "hello"

    def test_no_system_prompt_omits_system_role(self) -> None:
        seen = {}

        def responder(body, idx):
            seen["messages"] = body["messages"]
            return _FakeResponse(200, content="x")

        runner = _runner_with(responder, system="")
        runner.sample(["hello"], K=1, seed=0)
        assert [m["role"] for m in seen["messages"]] == ["user"]

    def test_max_tokens_and_temperature_forwarded(self) -> None:
        seen = {}

        def responder(body, idx):
            seen.update(body)
            return _FakeResponse(200, content="x")

        runner = _runner_with(responder)
        runner.sample(["q"], K=1, seed=0,
                      config=SamplingConfig(temperature=0.7, max_tokens=512))
        assert seen["max_tokens"] == 512
        assert seen["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:
    def test_k_must_be_positive(self) -> None:
        runner = _runner_with(lambda b, i: _FakeResponse(200, content="x"))
        with pytest.raises(ValueError, match="K must be"):
            runner.sample(["q"], K=0, seed=0)

    def test_greedy_with_k_gt_1_rejected(self) -> None:
        runner = _runner_with(lambda b, i: _FakeResponse(200, content="x"))
        with pytest.raises(ValueError, match="[Gg]reedy"):
            runner.sample(["q"], K=2, seed=0, config=SamplingConfig(temperature=0.0))


# ---------------------------------------------------------------------------
# Retry on transient status
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retries_then_succeeds(self, monkeypatch) -> None:
        # First call 429, second 200.
        import src.sampling.api_runner as mod
        monkeypatch.setattr(mod.asyncio, "sleep", _instant_sleep)
        state = {"n": 0}

        def responder(body, idx):
            state["n"] += 1
            if state["n"] == 1:
                return _FakeResponse(429, text="rate limited")
            return _FakeResponse(200, content="ok")

        runner = _runner_with(responder)
        out = runner.sample(["q"], K=1, seed=0)
        assert out == [["ok"]]
        assert state["n"] == 2

    def test_non_retryable_raises(self, monkeypatch) -> None:
        import src.sampling.api_runner as mod
        monkeypatch.setattr(mod.asyncio, "sleep", _instant_sleep)
        runner = _runner_with(lambda b, i: _FakeResponse(400, text="bad model"))
        with pytest.raises(RuntimeError):
            runner.sample(["q"], K=1, seed=0)


# ---------------------------------------------------------------------------
# Resume cache
# ---------------------------------------------------------------------------

class TestResume:
    def test_skips_cached_calls(self, tmp_path: Path) -> None:
        resume = tmp_path / "resume.jsonl"
        # Pre-seed (0,0) and (0,1); only (0,2) should hit the network.
        resume.write_text(
            json.dumps({"i": 0, "k": 0, "text": "cached0"}) + "\n"
            + json.dumps({"i": 0, "k": 1, "text": "cached1"}) + "\n"
        )
        n_calls = {"n": 0}

        def responder(body, idx):
            n_calls["n"] += 1
            return _FakeResponse(200, content="fresh")

        runner = _runner_with(responder, resume_path=resume)
        out = runner.sample(["q"], K=3, seed=0)
        assert out == [["cached0", "cached1", "fresh"]]
        assert n_calls["n"] == 1  # only the uncached (0,2)

    def test_appends_completed_calls(self, tmp_path: Path) -> None:
        resume = tmp_path / "resume.jsonl"
        runner = _runner_with(
            lambda b, i: _FakeResponse(200, content="z"), resume_path=resume,
        )
        runner.sample(["q0", "q1"], K=2, seed=0)
        lines = [json.loads(x) for x in resume.read_text().splitlines() if x.strip()]
        keys = {(r["i"], r["k"]) for r in lines}
        assert keys == {(0, 0), (0, 1), (1, 0), (1, 1)}


async def _instant_sleep(_):  # helper: skip backoff delays in tests
    return None
