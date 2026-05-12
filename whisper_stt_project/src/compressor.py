from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from .utils import ensure_dir, file_size_kb, signal_metrics

logger = logging.getLogger(__name__)


SUPPORTED_FORMATS = ("wav", "flac", "mp3", "ogg")


# ---------------------------------------------------------------------- #
# data containers
# ---------------------------------------------------------------------- #
@dataclass
class CompressionReport:
    source_path: str
    target_path: str
    target_format: str
    bitrate: str
    source_kb: float
    target_kb: float
    ratio: float
    rmse: float
    psnr_db: float
    lossless: bool

    def pretty(self) -> str:
        return (
            f"[{self.target_format.upper()}] "
            f"{self.source_kb:7.1f} kB -> {self.target_kb:7.1f} kB "
            f"(ratio {self.ratio:5.2f}x, "
            f"RMSE {self.rmse:.5f}, PSNR {self.psnr_db:6.2f} dB"
            f"{', LOSSLESS' if self.lossless else ''})"
        )


# ---------------------------------------------------------------------- #
# ffmpeg helpers
# ---------------------------------------------------------------------- #
def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_convert(src: Path, dst: Path,
                    bitrate: str | None = None,
                    extra: List[str] | None = None) -> None:
    """Call ffmpeg directly so we don't need pydub at runtime."""
    if not _have_ffmpeg():
        raise RuntimeError(
            "`ffmpeg` is not on PATH. Install it (apt/brew/choco) and retry."
        )
    cmd: List[str] = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if bitrate:
        cmd += ["-b:a", bitrate]
    if extra:
        cmd += extra
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------- #
# public API
# ---------------------------------------------------------------------- #
def compress(audio_path: str | Path,
             out_dir: str | Path,
             target_format: str = "mp3",
             bitrate: str = "64k") -> CompressionReport:
    """Compress *audio_path* into *target_format* and return a report.

    The decompressed round-trip file is written next to the compressed file
    with a ``_roundtrip.wav`` suffix so the caller can A/B-listen.
    """
    target_format = target_format.lower()
    if target_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported target format: {target_format}")

    src = Path(audio_path)
    out_dir = ensure_dir(out_dir)
    target = out_dir / f"{src.stem}.{target_format}"
    roundtrip = out_dir / f"{src.stem}_{target_format}_roundtrip.wav"

    # 1. compress
    if target_format == "wav":
        # "compressing" to WAV is just a copy — useful baseline
        shutil.copyfile(src, target)
    else:
        _ffmpeg_convert(src, target,
                        bitrate=bitrate if target_format in ("mp3", "ogg") else None)

    # 2. decompress back to PCM WAV for fidelity comparison
    _ffmpeg_convert(target, roundtrip)

    # 3. measure
    ref, _ = sf.read(str(src), always_2d=False)
    deg, _ = sf.read(str(roundtrip), always_2d=False)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    if deg.ndim > 1:
        deg = deg.mean(axis=1)

    rmse, psnr = signal_metrics(np.asarray(ref), np.asarray(deg))
    src_kb = file_size_kb(src)
    tgt_kb = file_size_kb(target)
    ratio = src_kb / tgt_kb if tgt_kb else float("inf")

    report = CompressionReport(
        source_path=str(src),
        target_path=str(target),
        target_format=target_format,
        bitrate=bitrate if target_format in ("mp3", "ogg") else "—",
        source_kb=src_kb,
        target_kb=tgt_kb,
        ratio=ratio,
        rmse=rmse,
        psnr_db=psnr,
        lossless=(target_format in ("wav", "flac")),
    )
    logger.info(report.pretty())
    return report


def benchmark(audio_path: str | Path,
              out_dir: str | Path,
              formats: tuple[str, ...] = ("flac", "mp3", "ogg"),
              mp3_bitrate: str = "64k",
              ogg_bitrate: str = "64k") -> Dict[str, CompressionReport]:
    """Run :func:`compress` for several formats and return a dict of reports."""
    reports: Dict[str, CompressionReport] = {}
    for fmt in formats:
        br = mp3_bitrate if fmt == "mp3" else (ogg_bitrate if fmt == "ogg" else "—")
        reports[fmt] = compress(audio_path, out_dir, target_format=fmt, bitrate=br)
    return reports


def reports_to_dicts(reports: Dict[str, CompressionReport]) -> List[dict]:
    return [asdict(r) for r in reports.values()]
