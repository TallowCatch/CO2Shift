"""Core package for reliable 4D CCS monitoring experiments."""

from __future__ import annotations

import sys
from pathlib import Path


def _maybe_enable_repo_vendor() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    vendor_root = repo_root / ".vendor"
    vendor_path = str(vendor_root)
    if vendor_root.exists() and vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


_maybe_enable_repo_vendor()

from .config import load_config
from .pipeline import run_all

__all__ = ["load_config", "run_all"]
