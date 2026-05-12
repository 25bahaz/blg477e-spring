import os
import sys
import json
import logging
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# Add project root to sys.path for local imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from whisper_stt_project.src.transcriber import WhisperTranscriber
    from whisper_stt_project.src.utils import ensure_dir
except ImportError:
    print("Error: Could not import whisper_stt_project modules.")
    print(f"Project root tried: {project_root}")
    sys.exit(1)

try:
    import jiwer
except ImportError:
    print("Error: 'jiwer' library is not installed. Please run 'pip install jiwer'.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for WER calculation (LibriSpeech style).
    
    Lowers case, removes punctuation, and collapses whitespace.
    """
    text = text.lower()
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_librispeech_transcripts(dataset_path: Path) -> Dict[str, str]:
    """Traverse LibriSpeech directory and map audio IDs to ground truth text."""
    transcripts = {}
    # LibriSpeech structure: test-clean/READER/CHAPTER/FILE.flac
    # Transcripts are in: test-clean/READER/CHAPTER/READER-CHAPTER.trans.txt
    
    trans_files = list(dataset_path.rglob("*.trans.txt"))
    logger.info(f"Found {len(trans_files)} transcript files.")
    
    for trans_file in trans_files:
        with open(trans_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    audio_id, text = parts
                    transcripts[audio_id] = normalize_text(text)
    
    return transcripts

def run_benchmark(
    librispeech_path: Path,
    models_to_test: List[str],
    limit: Optional[int] = None,
    device: Optional[str] = None
):
    """Run benchmarking for specified Whisper models."""
    
    # 1. Load ground truth
    logger.info(f"Loading ground truth transcripts from {librispeech_path}...")
    ground_truth_map = parse_librispeech_transcripts(librispeech_path)
    
    if not ground_truth_map:
        logger.error("No transcripts found. Check the LibriSpeech path.")
        return

    # 2. Find matching audio files
    audio_files = []
    for audio_id in ground_truth_map.keys():
        # IDs look like 1089-134686-0000
        parts = audio_id.split('-')
        reader_id, chapter_id = parts[0], parts[1]
        audio_path = librispeech_path / reader_id / chapter_id / f"{audio_id}.flac"
        
        if audio_path.exists():
            audio_files.append((audio_id, audio_path))
    
    if limit:
        audio_files = audio_files[:limit]
        logger.info(f"Limiting benchmark to {limit} samples.")
    
    logger.info(f"Ready to process {len(audio_files)} audio samples.")

    results = {}

    # 3. Test each model
    for model_name in models_to_test:
        transcriber = WhisperTranscriber(model_name=model_name, device=device, language="en")
        
        hypotheses = []
        references = []
        
        start_time = time.time()
        
        # Use tqdm for the main progress bar, ensuring it doesn't clash with other output
        pbar = tqdm(audio_files, desc=f"Model: {model_name}", unit="sample", leave=True)
        for audio_id, audio_path in pbar:
            try:
                # Transcribe (verbose=False is default, suppressing inner progress)
                res = transcriber.transcribe(audio_path)
                
                # Normalize and store
                hypotheses.append(normalize_text(res.text))
                references.append(ground_truth_map[audio_id])
                
            except Exception as e:
                # Use pbar.write to avoid breaking the progress bar
                pbar.write(f"Error transcribing {audio_id}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Calculate WER
        wer = jiwer.wer(references, hypotheses)
        
        results[model_name] = {
            "wer": wer,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(audio_files) if audio_files else 0,
            "samples_count": len(audio_files)
        }
        
        # Log final result for this model
        print(f"\n[DONE] {model_name} -> WER: {wer:.4f}, Total Time: {total_time:.2f}s")

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper models on LibriSpeech test-clean.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to LibriSpeech root")
    parser.add_argument("--models", nargs="+", default=["tiny", "base", "small"], help="Whisper models to test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per model")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file name")
    
    args = parser.parse_args()

    # Determine paths
    if args.data_dir:
        librispeech_path = Path(args.data_dir)
    else:
        # Try default locations
        librispeech_path = project_root / "LibriSpeech" / "test-clean"
        if not librispeech_path.exists():
            librispeech_path = project_root.parent / "LibriSpeech" / "test-clean"

    if not librispeech_path.exists():
        logger.error(f"LibriSpeech path not found: {librispeech_path}")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        librispeech_path=librispeech_path,
        models_to_test=args.models,
        limit=args.limit,
        device=args.device
    )

    if results:
        # Save results
        output_dir = project_root / "whisper_stt_project" / "outputs"
        ensure_dir(output_dir)
        output_file = output_dir / args.output
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Summary Table
        print("\n" + "="*50)
        print(f"{'Model':<10} | {'WER':<10} | {'Total Time (s)':<15}")
        print("-" * 50)
        for model, metrics in results.items():
            print(f"{model:<10} | {metrics['wer']:<10.4f} | {metrics['total_time_seconds']:<15.2f}")
        print("="*50)

if __name__ == "__main__":
    main()
