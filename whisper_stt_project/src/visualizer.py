"""Audio visualisation utilities.

Each ``plot_*`` function takes an audio file path (any format supported by
``librosa``) and writes a high-DPI PNG that is suitable for direct inclusion
in the LaTeX progress report.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# loading helper
# ---------------------------------------------------------------------- #
def load_audio(path: str | Path,
               sr: Optional[int] = 16_000,
               mono: bool = True) -> Tuple[np.ndarray, int]:
    """Wrapper around ``librosa.load`` that always returns ``float32`` data."""
    y, sr = librosa.load(str(path), sr=sr, mono=mono)
    return y.astype(np.float32), int(sr)


# ---------------------------------------------------------------------- #
# individual plots
# ---------------------------------------------------------------------- #
def plot_waveform(audio_path: str | Path,
                  out_path: str | Path,
                  sr: int = 16_000) -> Path:
    """Time-domain waveform."""
    y, sr = load_audio(audio_path, sr=sr)
    duration = len(y) / sr
    t = np.linspace(0, duration, num=len(y))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, y, linewidth=0.6, color="#1f77b4")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Waveform — {Path(audio_path).name}")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, duration)
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved waveform -> %s", out_path)
    return out_path


def plot_mel_spectrogram(audio_path: str | Path,
                         out_path: str | Path,
                         sr: int = 16_000,
                         n_mels: int = 128,
                         fmax: int = 8_000) -> Path:
    """Log-amplitude mel-spectrogram."""
    y, sr = load_audio(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    img = librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                   fmax=fmax, ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel-spectrogram — {Path(audio_path).name}")
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved mel-spectrogram -> %s", out_path)
    return out_path


def plot_mfcc(audio_path: str | Path,
              out_path: str | Path,
              sr: int = 16_000,
              n_mfcc: int = 20) -> Path:
    """MFCC (Mel-Frequency Cepstral Coefficients) heat-map."""
    y, sr = load_audio(audio_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax,
                                   cmap="viridis")
    fig.colorbar(img, ax=ax)
    ax.set_ylabel("MFCC index")
    ax.set_title(f"MFCC ({n_mfcc} coefficients) — {Path(audio_path).name}")
    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved MFCC -> %s", out_path)
    return out_path


def plot_zcr_and_energy(audio_path: str | Path,
                        out_path: str | Path,
                        sr: int = 16_000,
                        frame_length: int = 1024,
                        hop_length: int = 512) -> Path:
    """Zero-crossing-rate + short-time RMS energy in two subplots."""
    y, sr = load_audio(audio_path, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length,
                                             hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=frame_length,
                              hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr,
                                   hop_length=hop_length)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True)
    axes[0].plot(times, zcr, color="#d62728")
    axes[0].set_ylabel("ZCR")
    axes[0].set_title(f"Zero-crossing rate & RMS — {Path(audio_path).name}")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, rms, color="#2ca02c")
    axes[1].set_ylabel("RMS")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Saved ZCR/RMS -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------- #
# convenience: all four in one call
# ---------------------------------------------------------------------- #
def plot_all(audio_path: str | Path,
             out_dir: str | Path,
             sr: int = 16_000) -> dict[str, Path]:
    """Render the four standard plots into *out_dir* and return their paths."""
    out_dir = Path(out_dir)
    stem = Path(audio_path).stem
    return {
        "waveform":    plot_waveform(audio_path, out_dir / f"{stem}_waveform.png", sr=sr),
        "mel":         plot_mel_spectrogram(audio_path, out_dir / f"{stem}_mel.png", sr=sr),
        "mfcc":        plot_mfcc(audio_path, out_dir / f"{stem}_mfcc.png", sr=sr),
        "zcr_rms":     plot_zcr_and_energy(audio_path, out_dir / f"{stem}_zcr_rms.png", sr=sr),
    }
