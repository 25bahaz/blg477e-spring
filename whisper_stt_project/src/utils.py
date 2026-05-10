"""Small helpers shared across the pipeline modules."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create *path* (and parents) if it does not yet exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def file_size_kb(path: str | os.PathLike) -> float:
    """Return the size of *path* in kilobytes (1 kB = 1024 B)."""
    return os.path.getsize(path) / 1024.0


def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``HH:MM:SS,mmm`` for SRT output."""
    if seconds < 0:
        seconds = 0
    millis = int(round((seconds - int(seconds)) * 1000))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def signal_metrics(reference: np.ndarray,
                   degraded: np.ndarray) -> Tuple[float, float]:
    """Return ``(rmse, psnr_db)`` between two equally long signals.

    PSNR is computed against the peak amplitude of *reference*.  Both signals
    are truncated to the shorter length so a small round-trip mismatch in
    frame count does not raise.
    """
    n = min(len(reference), len(degraded))
    ref = reference[:n].astype(np.float64)
    deg = degraded[:n].astype(np.float64)
    err = ref - deg
    rmse = float(np.sqrt(np.mean(err ** 2)))
    peak = float(np.max(np.abs(ref))) if np.any(ref) else 1.0
    psnr = 20.0 * np.log10(peak / rmse) if rmse > 0 else float("inf")
    return rmse, psnr
