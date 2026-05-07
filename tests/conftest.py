"""Shared pytest fixtures for the rational-gap test suite.

Fixtures defined here are auto-discovered by pytest in any test
under ``tests/``. They centralise the small bits of test data that
multiple modules need so we have one source of truth.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from src.pipeline.cache import CacheKey
from src.pipeline.paths import Paths


@pytest.fixture
def paths_factory(tmp_path: Path) -> Callable[..., Paths]:
    """Return a factory that builds :class:`Paths` rooted in ``tmp_path``.

    Tests that exercise modules consuming a ``Paths`` (cache, logging)
    use this so they get an isolated, auto-cleaned filesystem layout.

    Usage::

        def test_x(paths_factory):
            paths = paths_factory()
            paths.ensure_dirs()
    """

    def _factory(**overrides: object) -> Paths:
        defaults: dict[str, Path] = {
            "outputs_root": tmp_path / "out",
            "samples_dir": tmp_path / "out" / "samples",
            "results_dir": tmp_path / "out" / "results",
            "raw_data_dir": tmp_path / "out" / "raw",
            "logs_dir": tmp_path / "out" / "logs",
        }
        defaults.update(overrides)  # type: ignore[arg-type]
        return Paths(**defaults)  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def cache_key_factory() -> Callable[..., CacheKey]:
    """Return a factory that builds :class:`CacheKey` with sensible defaults.

    Tests override only the fields they care about::

        def test_x(cache_key_factory):
            key = cache_key_factory(seed=42, K=8)
    """

    def _factory(**overrides: object) -> CacheKey:
        defaults: dict[str, object] = {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "dataset": "gsm8k",
            "K": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 1024,
            "seed": 42,
            "prompt_template_version": "v1",
        }
        defaults.update(overrides)
        return CacheKey(**defaults)  # type: ignore[arg-type]

    return _factory


@pytest.fixture
def humaneval_check_factory() -> Callable[..., str]:
    """Return a factory that builds a HumanEval check-program suffix.

    The suffix is what the HumanEval verifier appends *after* the
    model's generation: a ``def check(candidate)`` block plus the
    final ``check(<entry_point>)`` invocation.
    """

    def _build(asserts: str, entry_point: str = "candidate_func") -> str:
        return (
            f"def check(candidate):\n{textwrap.indent(asserts, '    ')}\n"
            f"check({entry_point})\n"
        )

    return _build


@pytest.fixture
def binary_utility_array() -> np.ndarray:
    """A small ``(M=4, K=4)`` binary utility array with a known gap.

    Per-prompt rational gaps:
        row 0: [0,0,0,0] → max=0, mean=0,    R=0
        row 1: [1,1,1,1] → max=1, mean=1,    R=0
        row 2: [1,0,0,0] → max=1, mean=0.25, R=0.75  (load-bearing case)
        row 3: [0,1,0,1] → max=1, mean=0.5,  R=0.5

    Aggregate: U_circ_K = 0.75, U_bar_K = 0.4375, R_hat_K = 0.3125.
    """
    return np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
        ],
        dtype=float,
    )
