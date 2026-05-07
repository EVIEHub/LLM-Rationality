"""Unit tests for logging utilities.

Covers run-logger setup (file path, idempotency on repeat calls,
DEBUG-vs-console level split), verifier audit log append semantics,
and compute-log append semantics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.pipeline.logging_utils import (
    log_compute,
    log_verifier_decision,
    setup_run_logger,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Reset root logger handlers around each test so tests don't
    inherit each other's file handles or duplicate output."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    for h in list(root.handlers):
        root.removeHandler(h)
    yield
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)
    for h in saved_handlers:
        root.addHandler(h)
    root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# setup_run_logger
# ---------------------------------------------------------------------------


def test_setup_run_logger_creates_file_in_runs_dir(tmp_path: Path) -> None:
    logger, log_file = setup_run_logger(tmp_path, "smoke", timestamp="20260507_120000Z")
    assert log_file == tmp_path / "runs" / "20260507_120000Z_smoke.log"
    assert log_file.parent.is_dir()
    assert log_file.exists()


def test_setup_run_logger_writes_log_lines(tmp_path: Path) -> None:
    _, log_file = setup_run_logger(tmp_path, "exp", timestamp="20260507_120000Z")
    log = logging.getLogger("test_module")
    log.info("starting")
    log.warning("anomaly noted")
    log.debug("per-prompt detail")
    # Force handlers to flush before reading.
    for h in logging.getLogger().handlers:
        h.flush()
    contents = log_file.read_text()
    assert "starting" in contents
    assert "anomaly noted" in contents
    assert "per-prompt detail" in contents  # DEBUG goes to file per AGENT.md §3.6


def test_setup_run_logger_repeat_call_does_not_accumulate_handlers(tmp_path: Path) -> None:
    setup_run_logger(tmp_path, "first", timestamp="20260507_120000Z")
    setup_run_logger(tmp_path, "second", timestamp="20260507_120001Z")
    root = logging.getLogger()
    assert len(root.handlers) == 2  # one file + one stream, not four


def test_setup_run_logger_console_level_is_independent_from_file_level(tmp_path: Path) -> None:
    setup_run_logger(tmp_path, "exp", timestamp="20260507_120000Z", console_level=logging.WARNING)
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    stream_handlers = [
        h for h in root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1
    assert len(stream_handlers) == 1
    assert file_handlers[0].level == logging.DEBUG
    assert stream_handlers[0].level == logging.WARNING


def test_setup_run_logger_default_timestamp_format(tmp_path: Path) -> None:
    """Default timestamps should match YYYYMMDD_HHMMSSZ."""
    import re
    _, log_file = setup_run_logger(tmp_path, "exp")
    name = log_file.name
    assert re.match(r"\d{8}_\d{6}Z_exp\.log", name), name


# ---------------------------------------------------------------------------
# log_verifier_decision
# ---------------------------------------------------------------------------


def test_log_verifier_decision_creates_per_dataset_file(tmp_path: Path) -> None:
    log_verifier_decision(tmp_path, "gsm8k", {"prompt_id": "p1", "utility": 1.0})
    f = tmp_path / "verifier" / "gsm8k_log.jsonl"
    assert f.exists()
    line = f.read_text().strip()
    assert json.loads(line) == {"prompt_id": "p1", "utility": 1.0}


def test_log_verifier_decision_appends_multiple_lines(tmp_path: Path) -> None:
    for i in range(3):
        log_verifier_decision(tmp_path, "math", {"prompt_id": f"p{i}", "utility": float(i % 2)})
    f = tmp_path / "verifier" / "math_log.jsonl"
    lines = [json.loads(l) for l in f.read_text().splitlines()]
    assert len(lines) == 3
    assert lines[0]["prompt_id"] == "p0"
    assert lines[2]["prompt_id"] == "p2"


def test_log_verifier_decision_separates_datasets(tmp_path: Path) -> None:
    log_verifier_decision(tmp_path, "gsm8k", {"x": 1})
    log_verifier_decision(tmp_path, "math", {"x": 2})
    log_verifier_decision(tmp_path, "humaneval", {"x": 3})
    files = sorted((tmp_path / "verifier").iterdir())
    assert [f.name for f in files] == [
        "gsm8k_log.jsonl",
        "humaneval_log.jsonl",
        "math_log.jsonl",
    ]


# ---------------------------------------------------------------------------
# log_compute
# ---------------------------------------------------------------------------


def test_log_compute_creates_jsonl(tmp_path: Path) -> None:
    log_compute(tmp_path, experiment="h1_smoke", gpu_hours=0.5)
    f = tmp_path / "compute_budget.jsonl"
    assert f.exists()
    record = json.loads(f.read_text().strip())
    assert record["experiment"] == "h1_smoke"
    assert record["gpu_hours"] == 0.5
    assert "timestamp" in record


def test_log_compute_appends_records(tmp_path: Path) -> None:
    log_compute(tmp_path, experiment="h1", gpu_hours=1.0, timestamp="2026-05-07T12:00:00+00:00")
    log_compute(tmp_path, experiment="h2", gpu_hours=2.5, timestamp="2026-05-07T13:00:00+00:00")
    log_compute(tmp_path, experiment="h3", gpu_hours=0.25, timestamp="2026-05-07T14:00:00+00:00")
    lines = [json.loads(l) for l in (tmp_path / "compute_budget.jsonl").read_text().splitlines()]
    assert len(lines) == 3
    assert sum(l["gpu_hours"] for l in lines) == 3.75


def test_log_compute_carries_metadata(tmp_path: Path) -> None:
    log_compute(
        tmp_path,
        experiment="h1",
        gpu_hours=1.0,
        metadata={"model": "Qwen-1.5B", "num_prompts": 10, "K": 4},
    )
    record = json.loads((tmp_path / "compute_budget.jsonl").read_text().strip())
    assert record["metadata"] == {"model": "Qwen-1.5B", "num_prompts": 10, "K": 4}


def test_log_compute_default_timestamp_is_iso8601(tmp_path: Path) -> None:
    from datetime import datetime
    log_compute(tmp_path, experiment="h1", gpu_hours=1.0)
    record = json.loads((tmp_path / "compute_budget.jsonl").read_text().strip())
    # Should be parseable as ISO 8601.
    parsed = datetime.fromisoformat(record["timestamp"])
    assert parsed.tzinfo is not None  # UTC offset attached
