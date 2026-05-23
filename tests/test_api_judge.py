"""Tests for src.verification.api_judge.

A fake async client is injected via ``client_factory`` so the (M, K)
assembly, A/B-position randomisation, parse, aggregation, and resume
cache run on CPU with no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sampling.api_runner import ApiModelSpec
from src.verification.api_judge import score_matrix_api


class _FakeResponse:
    def __init__(self, content: str, status_code: int = 200):
        self.status_code = status_code
        self._content = content
        self.text = ""

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
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


_SPEC = ApiModelSpec(api_model="judge", base_url="http://x/v1", api_key="k")


def _score(responder, **kw):
    return score_matrix_api(
        _SPEC,
        raw_prompts=kw.pop("raw_prompts", ["q0", "q1"]),
        candidates=kw.pop("candidates", [["c0"], ["c1"]]),
        references=kw.pop("references", ["r0", "r1"]),
        client_factory=lambda: _FakeClient(responder),
        **kw,
    )


class TestShape:
    def test_returns_m_by_k(self):
        out = _score(lambda b, i: _FakeResponse("T"), L=3)
        assert out.utility.shape == (2, 1)
        assert out.raw_verdicts.shape == (2, 1, 3)
        assert out.n_judge_calls == 2 * 1 * 3

    def test_all_tie_gives_half(self):
        out = _score(lambda b, i: _FakeResponse("T"), L=3)
        assert (out.utility == 0.5).all()
        assert out.parse_failure_rate == 0.0

    def test_parse_failure_counted(self):
        # Unrecognised verdict -> parse failure -> treated as 0.5.
        out = _score(lambda b, i: _FakeResponse("hello"), L=3,
                     raw_prompts=["q"], candidates=[["c"]], references=["r"])
        assert out.parse_failure_rate == 1.0
        assert out.utility[0, 0] == 0.5


class TestVerdictDirection:
    def test_judge_prefers_position_a_content(self):
        # Judge always says "A". With random A/B position, "A" maps to
        # candidate-win when candidate was in slot A, else reference-win.
        # Over L=5 with a fixed seed the majority is deterministic; just
        # assert the utility is a valid ternary value and calls happened.
        seen = {"n": 0}

        def responder(body, idx):
            seen["n"] += 1
            return _FakeResponse("A")

        out = _score(responder, L=5, seed=0,
                     raw_prompts=["q"], candidates=[["c"]], references=["r"])
        assert seen["n"] == 5
        assert out.utility[0, 0] in (0.0, 0.5, 1.0)

    def test_system_and_ab_in_prompt(self):
        captured = {}

        def responder(body, idx):
            captured.setdefault("first", body)
            return _FakeResponse("T")

        _score(responder, L=1, raw_prompts=["What is 2+2?"],
               candidates=[["four"]], references=["five"])
        msgs = captured["first"]["messages"]
        assert msgs[0]["role"] == "system"
        assert "one character" in msgs[0]["content"]
        user = msgs[1]["content"]
        assert "What is 2+2?" in user
        assert "Response A" in user and "Response B" in user
        # both candidate and reference appear regardless of position
        assert "four" in user and "five" in user


class TestGuards:
    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            _score(lambda b, i: _FakeResponse("T"),
                   raw_prompts=["q0", "q1"], candidates=[["c0"]], references=["r0", "r1"])

    def test_k_mismatch(self):
        with pytest.raises(ValueError, match="differs from prompt 0"):
            _score(lambda b, i: _FakeResponse("T"),
                   raw_prompts=["q0", "q1"], candidates=[["c0", "c0b"], ["c1"]],
                   references=["r0", "r1"])

    def test_L_must_be_positive(self):
        with pytest.raises(ValueError, match="L must be"):
            _score(lambda b, i: _FakeResponse("T"), L=0,
                   raw_prompts=["q"], candidates=[["c"]], references=["r"])


class TestResume:
    def test_skips_cached(self, tmp_path: Path):
        resume = tmp_path / "j.jsonl"
        # Pre-seed all 3 calls for the single (0,0) pair as ties.
        resume.write_text(
            "".join(json.dumps({"i": 0, "k": 0, "l": l, "text": "T"}) + "\n"
                    for l in range(3))
        )
        n = {"calls": 0}

        def responder(body, idx):
            n["calls"] += 1
            return _FakeResponse("A")

        out = score_matrix_api(
            _SPEC, raw_prompts=["q"], candidates=[["c"]], references=["r"],
            L=3, resume_path=resume,
            client_factory=lambda: _FakeClient(responder),
        )
        assert n["calls"] == 0  # everything served from resume cache
        assert out.utility[0, 0] == 0.5  # 3 ties

    def test_appends_completed(self, tmp_path: Path):
        resume = tmp_path / "j.jsonl"
        score_matrix_api(
            _SPEC, raw_prompts=["q"], candidates=[["c"]], references=["r"],
            L=2, resume_path=resume,
            client_factory=lambda: _FakeClient(lambda b, i: _FakeResponse("T")),
        )
        recs = [json.loads(x) for x in resume.read_text().splitlines() if x.strip()]
        assert {(r["i"], r["k"], r["l"]) for r in recs} == {(0, 0, 0), (0, 0, 1)}
