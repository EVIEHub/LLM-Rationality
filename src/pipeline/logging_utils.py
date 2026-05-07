"""Logging utilities for the rational-gap pipeline.

Three categories of logs are written, per AGENT.md §3.3 and §3.6:

1. **Run logs** (``${logs_dir}/runs/<timestamp>_<experiment>.log``):
   human-readable Python-logging output configured once per entry
   point via :func:`setup_run_logger`.

2. **Verifier audit logs** (``${logs_dir}/verifier/{dataset}_log.jsonl``):
   structured JSONL of every verifier decision (input, ground truth,
   output). Appended via :func:`log_verifier_decision`. Required by
   AGENT.md §3.3 — paper reviewers must be able to audit every
   verifier decision after the fact.

3. **Compute log** (``${logs_dir}/compute_budget.jsonl``): cumulative
   GPU-hour budget appended via :func:`log_compute` at the end of
   each run. Reported in the paper.

All write paths use simple append / fresh-file modes — no rotation,
no buffering complications. The file is the source of truth.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RUN_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_run_logger(
    logs_dir: Path,
    experiment: str,
    *,
    console_level: int = logging.INFO,
    timestamp: str | None = None,
) -> tuple[logging.Logger, Path]:
    """Configure the root logger for one pipeline run.

    Replaces any existing root-logger handlers with a fresh pair: a
    ``FileHandler`` writing DEBUG-level output to
    ``${logs_dir}/runs/<timestamp>_<experiment>.log`` and a
    ``StreamHandler`` writing ``console_level`` and above to stderr.

    Subsequent module-level loggers obtained via
    ``logging.getLogger(__name__)`` automatically flow into both.

    Args:
        logs_dir: Logs root (typically ``Paths.logs_dir``).
        experiment: Short identifier such as ``"h1_qwen_gsm8k"``.
        console_level: Minimum level shown on stderr; the file always
            captures DEBUG and above per AGENT.md §3.6.
        timestamp: ISO-style timestamp injected into the filename.
            Defaults to ``YYYYMMDD_HHMMSSZ`` in UTC.

    Returns:
        Tuple of (root logger, absolute log file path).
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    runs_dir = logs_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_file = runs_dir / f"{timestamp}_{experiment}.log"

    formatter = logging.Formatter(_RUN_LOG_FORMAT)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_level)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    # Close and remove any pre-existing handlers so repeated calls don't
    # accumulate file descriptors or duplicate log lines.
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.setLevel(logging.DEBUG)

    return root, log_file


def log_verifier_decision(
    logs_dir: Path,
    dataset: str,
    record: dict[str, Any],
) -> None:
    """Append a single verifier-decision record to the per-dataset audit log.

    Per AGENT.md §3.3, every verifier decision must be logged.
    Skipping these records "for performance" is forbidden — the audit
    trail is a paper-review requirement.

    Args:
        logs_dir: Logs root.
        dataset: Dataset name; determines the filename
            (``${logs_dir}/verifier/{dataset}_log.jsonl``).
        record: A JSON-serialisable dict. Recommended keys:
            ``prompt_id``, ``generation``, ``ground_truth``,
            ``utility``, ``timestamp``. The exact schema is the
            caller's choice — this function just appends.
    """
    verifier_dir = logs_dir / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    log_file = verifier_dir / f"{dataset}_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def log_compute(
    logs_dir: Path,
    *,
    experiment: str,
    gpu_hours: float,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> None:
    """Append a compute-budget entry.

    The cumulative budget is the sum of all ``gpu_hours`` entries in
    ``${logs_dir}/compute_budget.jsonl`` across all runs; the paper
    reports it as a single number.

    Args:
        logs_dir: Logs root.
        experiment: Experiment identifier (e.g. ``"h2_tulu3_dpo_math"``).
        gpu_hours: Hours spent in this run / step. Caller chooses
            whether entries are incremental (recommended — sum gives
            cumulative) or already cumulative (then the paper reports
            the latest entry).
        metadata: Free-form dict for additional context (model,
            num_prompts, K, hardware, etc.).
        timestamp: ISO-8601 timestamp; default is now in UTC.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "compute_budget.jsonl"

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    record: dict[str, Any] = {
        "timestamp": timestamp,
        "experiment": experiment,
        "gpu_hours": gpu_hours,
    }
    if metadata is not None:
        record["metadata"] = metadata

    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
