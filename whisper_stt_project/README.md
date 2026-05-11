# Whisper Speech-to-Text Pipeline

A multimedia-computing course project that combines OpenAI's **Whisper** speech-to-text
model with classical audio-DSP visualisations and lossy/lossless compression
experiments.

## Features

1. **Speech-to-text transcription** with Whisper (any model size, language
   auto-detection, segment-level timestamps, plain-text and SRT export).
2. **Live microphone listening** — real-time transcription directly from your mic.
3. **Audio visualisation** — waveform, mel-spectrogram, MFCC heatmap and
   zero-crossing-rate plot.
4. **Compression / decompression** between WAV, FLAC, MP3 and OGG with
   compression-ratio reporting and round-trip quality (PSNR, RMSE) checks.
5. A single CLI driver (`src/main.py`) ties everything together and produces
   ready-to-include figures for the project report.

## Layout

```
whisper_stt_project/
├── requirements.txt
├── README.md
├── src/
│   ├── main.py            # CLI entry point
│   ├── transcriber.py     # Whisper wrapper
│   ├── listener.py        # Real-time microphone capture
│   ├── visualizer.py      # waveform / spectrogram / MFCC plots
│   ├── compressor.py      # WAV ↔ MP3/FLAC/OGG with metrics
│   └── utils.py
├── samples/               # input audio (place .wav/.mp3 files here)
├── figures/               # generated plots
├── outputs/               # transcripts and compressed/decompressed audio
└── report/                # LaTeX progress report
```

## Quick start

```bash
pip install -r requirements.txt          # also requires ffmpeg on PATH
python src/main.py demo                   # run the full demo on synthetic audio
python src/main.py listen --model tiny    # live transcription from microphone
python src/main.py transcribe samples/speech.wav --model base --srt
python src/main.py visualize  samples/speech.wav
python src/main.py compress   samples/speech.wav --target mp3 --bitrate 64k
```

## Authors

`[Your Name]` — `[Student ID]`
Course: `[Course Code – Multimedia Computing]`
Instructor: Ali Altan
