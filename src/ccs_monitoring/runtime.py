"""Runtime helpers for repeatable local execution."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np


def ensure_runtime_environment(output_root: str | Path, seed: int) -> None:
    """Set safe environment defaults for this local machine."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mpl_dir = output_root / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)


def ensure_torch_seed(seed: int) -> None:
    """Seed torch lazily so importing this module does not force torch import."""
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
