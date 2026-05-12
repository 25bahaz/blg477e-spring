from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Allow running both as ``python -m src.main`` and ``python src/main.py``
if __package__ in (None, ""):
    THIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(THIS_DIR.parent))
    from src import compressor, listener, transcriber, visualizer
    from src.utils import ensure_dir
else:                                                       # pragma: no cover
    from . import compressor, listener, transcriber, visualizer
    from .utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _make_synthetic_speech(out_path: Path,
                           duration: float = 5.0,
                           sr: int = 16_000) -> Path:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    chirp = 0.4 * np.sin(2 * np.pi * (200 + 500 * t / duration) * t)
    tone = 0.2 * np.sin(2 * np.pi * 880 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    noise = 0.02 * np.random.randn(len(t))
    y = (chirp + tone + noise).astype(np.float32)
    y = np.clip(y, -1.0, 1.0)
    ensure_dir(out_path.parent)
    sf.write(str(out_path), y, sr, subtype="PCM_16")
    return out_path


# ---------------------------------------------------------------------- #
# sub-command implementations
# ---------------------------------------------------------------------- #
def cmd_transcribe(args: argparse.Namespace) -> None:
    audio = Path(args.audio).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    t = transcriber.WhisperTranscriber(
        model_name=args.model, device=args.device, language=args.language)
    result = t.transcribe(audio, verbose=args.verbose)

    txt = transcriber.WhisperTranscriber.write_txt(
        result, out_dir / f"{audio.stem}.txt")
    js = transcriber.WhisperTranscriber.write_json(
        result, out_dir / f"{audio.stem}.json")
    print(f"Detected language : {result.language}")
    print(f"Transcript        : {txt}")
    print(f"JSON              : {js}")
    if args.srt:
        srt = transcriber.WhisperTranscriber.write_srt(
            result, out_dir / f"{audio.stem}.srt")
        print(f"SubRip            : {srt}")
    print("---- TEXT ----")
    print(result.text)


def cmd_listen(args: argparse.Namespace) -> None:
    print(f"Loading Whisper model '{args.model}'...")
    t = transcriber.WhisperTranscriber(
        model_name=args.model, device=args.device, language=args.language)
    
    print(f"Listening... (Press Ctrl+C to stop)")
    print("-" * 30)
    
    try:
        with listener.MicrophoneListener(sample_rate=16_000) as mic:
            for chunk in mic.listen(chunk_duration=args.chunk_size):
                result = t.transcribe(chunk)
                if result.text.strip():
                    print(f"[{result.language}] {result.text}")
    except KeyboardInterrupt:
        print("\nStopped by user.")


def cmd_visualize(args: argparse.Namespace) -> None:
    audio = Path(args.audio).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    paths = visualizer.plot_all(audio, out_dir, sr=args.sample_rate)
    for k, p in paths.items():
        print(f"{k:10s} -> {p}")


def cmd_compress(args: argparse.Namespace) -> None:
    audio = Path(args.audio).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    report = compressor.compress(
        audio, out_dir, target_format=args.target, bitrate=args.bitrate)
    print(report.pretty())


def cmd_benchmark(args: argparse.Namespace) -> None:
    audio = Path(args.audio).expanduser().resolve()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    reports = compressor.benchmark(audio, out_dir)
    print(f"\nBenchmark for {audio.name}")
    print("-" * 70)
    for r in reports.values():
        print(r.pretty())
    if args.json:
        out = out_dir / f"{audio.stem}_compression.json"
        out.write_text(json.dumps(compressor.reports_to_dicts(reports),
                                  indent=2), encoding="utf-8")
        print(f"\nReport JSON -> {out}")


def cmd_demo(args: argparse.Namespace) -> None:
    samples_dir = ensure_dir(PROJECT_ROOT / "samples")
    figures_dir = ensure_dir(PROJECT_ROOT / "figures")
    outputs_dir = ensure_dir(PROJECT_ROOT / "outputs")

    audio_path = samples_dir / "synthetic_demo.wav"
    print(f"[1/3] Synthesising test signal -> {audio_path}")
    _make_synthetic_speech(audio_path, duration=args.duration)

    print("[2/3] Generating visualisations ...")
    paths = visualizer.plot_all(audio_path, figures_dir)
    for k, p in paths.items():
        print(f"      {k:10s} -> {p}")

    if args.skip_compression:
        print("[3/3] Compression benchmark skipped (--skip-compression).")
    else:
        print("[3/3] Running compression benchmark ...")
        try:
            reports = compressor.benchmark(audio_path, outputs_dir)
            for r in reports.values():
                print("      " + r.pretty())
            (outputs_dir / "compression.json").write_text(
                json.dumps(compressor.reports_to_dicts(reports), indent=2),
                encoding="utf-8")
        except RuntimeError as e:
            print(f"      Compression skipped: {e}")

    if args.transcribe:
        print("[*] Running Whisper on the synthetic signal "
              "(output will be gibberish — sanity check only) ...")
        t = transcriber.WhisperTranscriber(model_name=args.model)
        res = t.transcribe(audio_path, verbose=False)
        print(f"      detected language: {res.language}")
        print(f"      text: {res.text!r}")


# ---------------------------------------------------------------------- #
# argument parser
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whisper-stt",
        description="Whisper-based speech-to-text + visualisation + "
                    "compression pipeline (multimedia-computing project).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    # transcribe ------------------------------------------------------- #
    pt = sub.add_parser("transcribe", help="Run Whisper on an audio file")
    pt.add_argument("audio")
    pt.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium", "large",
                             "large-v2", "large-v3"])
    pt.add_argument("--language", default=None,
                    help="ISO-639-1 code, e.g. 'en'. None = auto-detect.")
    pt.add_argument("--device", default=None, help="'cpu' / 'cuda' / None")
    pt.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"))
    pt.add_argument("--srt", action="store_true",
                    help="Also write SubRip .srt subtitles")
    pt.set_defaults(func=cmd_transcribe)

    # visualize -------------------------------------------------------- #
    pv = sub.add_parser("visualize",
                        help="Render waveform / spectrogram / MFCC / ZCR")
    pv.add_argument("audio")
    pv.add_argument("--out-dir", default=str(PROJECT_ROOT / "figures"))
    pv.add_argument("--sample-rate", type=int, default=16_000)
    pv.set_defaults(func=cmd_visualize)

    # compress --------------------------------------------------------- #
    pc = sub.add_parser("compress", help="Compress to a single target format")
    pc.add_argument("audio")
    pc.add_argument("--target", choices=compressor.SUPPORTED_FORMATS,
                    default="mp3")
    pc.add_argument("--bitrate", default="64k",
                    help="Bitrate for lossy formats (e.g. 64k, 128k)")
    pc.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"))
    pc.set_defaults(func=cmd_compress)

    # benchmark -------------------------------------------------------- #
    pb = sub.add_parser("benchmark",
                        help="Compress to FLAC, MP3 and OGG for comparison")
    pb.add_argument("audio")
    pb.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs"))
    pb.add_argument("--json", action="store_true",
                    help="Also dump a JSON report next to the audio.")
    pb.set_defaults(func=cmd_benchmark)

    # demo ------------------------------------------------------------- #
    pd = sub.add_parser("demo",
                        help="Synthesise a test signal and run everything.")
    pd.add_argument("--duration", type=float, default=5.0)
    pd.add_argument("--skip-compression", action="store_true",
                    help="Skip ffmpeg-dependent compression step.")
    pd.add_argument("--transcribe", action="store_true",
                    help="Also run Whisper on the synthetic clip.")
    pd.add_argument("--model", default="base")
    pd.set_defaults(func=cmd_demo)

    # listen ----------------------------------------------------------- #
    pl = sub.add_parser("listen", help="Transcribe live audio from microphone")
    pl.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium", "large",
                             "large-v2", "large-v3"])
    pl.add_argument("--language", default=None,
                    help="ISO-639-1 code, e.g. 'en'. None = auto-detect.")
    pl.add_argument("--device", default=None, help="'cpu' / 'cuda' / None")
    pl.add_argument("--chunk-size", type=float, default=3.0,
                    help="Duration of each audio chunk to transcribe (seconds)")
    pl.set_defaults(func=cmd_listen)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
