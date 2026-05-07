"""Verification subsystem: unified registry over per-dataset verifiers.

Public API re-exported from :mod:`src.verification.interface`.
"""

from src.verification.interface import (
    Verifier,
    get_verifier,
    known_datasets,
    verify,
)

__all__ = ["Verifier", "get_verifier", "known_datasets", "verify"]
