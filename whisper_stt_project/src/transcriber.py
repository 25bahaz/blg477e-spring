"""Whisper-based speech-to-text wrapper.

The class :class:`WhisperTranscriber` lazily loads a Whisper model the first
time :meth:`transcribe` is called.  It returns the raw Whisper result *and*
provides convenience methods for writing plain-text and SubRip (``.srt``)
files for the project demos.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import ensure_dir, format_timestamp

logger = logging.getLogger(__name__)


# A small allow-list keeps the CLI honest; "auto" lets Whisper detect.
_VALID_MODELS = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}


@dataclass
class TranscriptionResult:
    """Light-weight, JSON-serialisable view over Whisper's raw output."""

    text: str
    language: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "language": self.language,
            "text": self.text,
            "segments": [
                {
                    "id": s.get("id"),
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": s.get("text", "").strip(),
                }
                for s in self.segments
            ],
        }


class WhisperTranscriber:
    """Thin wrapper around ``openai-whisper``.

    Parameters
    ----------
    model_name:
        One of ``tiny``, ``base``, ``small``, ``medium``, ``large``,
        ``large-v2``, ``large-v3``.
    device:
        ``"cuda"``, ``"cpu"`` or ``None`` for auto-detect.
    language:
        ISO-639-1 code (e.g. ``"en"``, ``"tr"``) or ``None`` to let Whisper
        auto-detect from the first 30 seconds.
    """

    def __init__(self,
                 model_name: str = "base",
                 device: Optional[str] = None,
                 language: Optional[str] = None) -> None:
        if model_name not in _VALID_MODELS:
            raise ValueError(
                f"Unknown Whisper model {model_name!r}. "
                f"Choose one of: {sorted(_VALID_MODELS)}"
            )
        self.model_name = model_name
        self.device = device
        self.language = language
        self._model = None  # lazy

    # ------------------------------------------------------------------ #
    # core
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        """Import + instantiate Whisper on first use."""
        if self._model is not None:
            return
        try:
            import whisper  # imported lazily so the rest of the project
                            # works without torch installed.
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai-whisper is not installed. Run "
                "`pip install -r requirements.txt`."
            ) from e

        logger.info("Loading Whisper model %s ...", self.model_name)
        self._model = whisper.load_model(self.model_name, device=self.device)
        logger.info("Whisper model %s ready.", self.model_name)

    def transcribe(self,
                   audio_path: str | Path,
                   verbose: bool = False) -> TranscriptionResult:
        """Run Whisper on *audio_path* and return a :class:`TranscriptionResult`."""
        self._load()
        audio_path = str(audio_path)
        logger.info("Transcribing %s ...", audio_path)

        result = self._model.transcribe(
            audio_path,
            language=self.language,
            verbose=verbose,
            fp16=False,  # safe default on CPU
        )
        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", "?"),
            segments=result.get("segments", []),
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------ #
    # writers
    # ------------------------------------------------------------------ #
    @staticmethod
    def write_txt(result: TranscriptionResult, path: str | Path) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(result.text + "\n", encoding="utf-8")
        return path

    @staticmethod
    def write_json(result: TranscriptionResult, path: str | Path) -> Path:
        path = Path(path)
        ensure_dir(path.parent)
        path.write_text(json.dumps(result.to_dict(), indent=2,
                                   ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def write_srt(result: TranscriptionResult, path: str | Path) -> Path:
        """Write segments as a SubRip (``.srt``) subtitle file."""
        path = Path(path)
        ensure_dir(path.parent)
        lines: List[str] = []
        for i, seg in enumerate(result.segments, start=1):
            start = format_timestamp(float(seg.get("start", 0.0)))
            end = format_timestamp(float(seg.get("end", 0.0)))
            text = seg.get("text", "").strip()
            lines.extend([str(i), f"{start} --> {end}", text, ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
