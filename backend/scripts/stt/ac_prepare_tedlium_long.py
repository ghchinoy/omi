"""
TED-LIUM 3 Long-Form Dataset Preparation

Downloads and extracts a stratified two-tier subset of full TED talks from
Hugging Face `distil-whisper/tedlium-long-form`, standardizes audio to 16 kHz
mono 16-bit PCM WAV format, normalizes reference transcripts for WER evaluation,
and outputs audio files alongside `manifest.json` and `manifest.jsonl`.

Tier A (Medium-Long): ~3 to 6 minute talks (Batch + Streaming evaluation)
Tier B (Long):        ~13 to 19 minute talks (Batch-only evaluation)

Usage:
    uv run --python 3.11 --with datasets --with soundfile --with pyarrow \
        python3 scripts/stt/ac_prepare_tedlium_long.py
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Default output directory in repo
REPO_ROOT = Path('/Users/ghchinoy/projects/omi-dev')
DEFAULT_OUT_DIR = REPO_ROOT / 'benchmarks/data/tedlium_long'

TIER_A_SPEAKERS = [
    'Brian_Cox',        # ~187s (3.1m), 590 words
    'Al_Gore',          # ~291s (4.9m), 879 words
    'David_Merrill',    # ~336s (5.6m), 1108 words
    'RobertGupta',      # ~340s (5.7m), 876 words
]

TIER_B_SPEAKERS = [
    'DanBarber',        # ~834s (13.9m), 2383 words
    'Craig_Venter',     # ~908s (15.1m), 2398 words
    'MichaelSpecter',   # ~921s (15.4m), 2966 words
    'Barry_Schwartz',   # ~1105s (18.4m), 3219 words
]


def normalize_tedlium_text(text: str) -> str:
    """
    Normalizes raw TED-LIUM reference text:
    - Replaces spaced apostrophes (e.g. "it 's" -> "it's")
    - Collapses whitespace
    - Uppercases for consistency with LibriSpeech manifest format
    """
    cleaned = text.replace(" '", "'")
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned.upper()


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare TED-LIUM 3 Long-Form benchmark dataset')
    parser.add_argument('--out-dir', type=str, default=str(DEFAULT_OUT_DIR), help='Destination directory')
    parser.add_argument('--force', action='store_true', help='Force re-download and re-transcode')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_json_path = out_dir / 'manifest.json'
    manifest_jsonl_path = out_dir / 'manifest.jsonl'

    if manifest_json_path.exists() and not args.force:
        print(f"Manifest already exists at: {manifest_json_path}")
        print("Use --force to overwrite and re-prepare.")
        return

    print("Loading Hugging Face dataset 'distil-whisper/tedlium-long-form'...")
    try:
        from datasets import Audio, load_dataset
        import soundfile as sf
    except ImportError as e:
        print(f"ERROR: Missing required dependencies ({e}).")
        print("Please run with:")
        print("  uv run --python 3.11 --with datasets --with soundfile --with pyarrow python3 scripts/stt/ac_prepare_tedlium_long.py")
        sys.exit(1)

    # Collect talk records from validation and test splits
    talks: Dict[str, Dict[str, Any]] = {}
    for split in ['validation', 'test']:
        print(f"Loading '{split}' split...")
        ds = load_dataset('distil-whisper/tedlium-long-form', split=split)
        ds = ds.cast_column('audio', Audio(decode=False))
        for row in ds:
            spk = row['speaker_id']
            talks[spk] = {
                'speaker': spk,
                'raw_bytes': row['audio']['bytes'],
                'text': row['text'],
                'split': split,
            }

    print(f"Discovered {len(talks)} total talks across splits.")

    manifest_entries: List[Dict[str, Any]] = []

    # 1. Process Tier A
    print("\n--- Processing Tier A (Medium-Long: ~3 to 6 min) ---")
    for i, spk in enumerate(TIER_A_SPEAKERS):
        if spk not in talks:
            print(f"Warning: Speaker {spk} not found in dataset, skipping.")
            continue
        
        sample_id = f"sample_a{i + 1:02d}"
        talk_data = talks[spk]
        wav_path = out_dir / f"{sample_id}.wav"

        # Transcode raw audio bytes to 16kHz mono PCM16 WAV
        temp_input = out_dir / f"{sample_id}_raw.wav"
        temp_input.write_bytes(talk_data['raw_bytes'])

        cmd = [
            'ffmpeg', '-y', '-i', str(temp_input),
            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
            str(wav_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if temp_input.exists():
            temp_input.unlink()

        # Probe duration
        info = sf.info(str(wav_path))
        duration_s = round(info.duration, 2)
        norm_text = normalize_tedlium_text(talk_data['text'])
        word_count = len(norm_text.split())
        size_kb = round(wav_path.stat().st_size / 1024, 1)

        entry = {
            'id': sample_id,
            'uid': f"tedlium-{spk}",
            'speaker': spk,
            'tier': 'a',
            'text': norm_text,
            'description': f"Tier A (Medium-Long): {spk} TED talk ({duration_s:.1f}s, {word_count} words)",
            'word_count': word_count,
            'duration_s': duration_s,
            'size_kb': size_kb,
        }
        manifest_entries.append(entry)
        print(f"  [{sample_id}] {spk:<20} {duration_s:6.1f}s ({duration_s/60:4.1f}m)  {word_count:5d} words  ({size_kb} KB)")

    # 2. Process Tier B
    print("\n--- Processing Tier B (Full Long-Form: ~13 to 19 min) ---")
    for i, spk in enumerate(TIER_B_SPEAKERS):
        if spk not in talks:
            print(f"Warning: Speaker {spk} not found in dataset, skipping.")
            continue
        
        sample_id = f"sample_b{i + 1:02d}"
        talk_data = talks[spk]
        wav_path = out_dir / f"{sample_id}.wav"

        # Transcode raw audio bytes to 16kHz mono PCM16 WAV
        temp_input = out_dir / f"{sample_id}_raw.wav"
        temp_input.write_bytes(talk_data['raw_bytes'])

        cmd = [
            'ffmpeg', '-y', '-i', str(temp_input),
            '-ar', '16000', '-ac', '1', '-sample_fmt', 's16',
            str(wav_path)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if temp_input.exists():
            temp_input.unlink()

        # Probe duration
        info = sf.info(str(wav_path))
        duration_s = round(info.duration, 2)
        norm_text = normalize_tedlium_text(talk_data['text'])
        word_count = len(norm_text.split())
        size_kb = round(wav_path.stat().st_size / 1024, 1)

        entry = {
            'id': sample_id,
            'uid': f"tedlium-{spk}",
            'speaker': spk,
            'tier': 'b',
            'text': norm_text,
            'description': f"Tier B (Full Long-Form): {spk} TED talk ({duration_s:.1f}s, {word_count} words)",
            'word_count': word_count,
            'duration_s': duration_s,
            'size_kb': size_kb,
        }
        manifest_entries.append(entry)
        print(f"  [{sample_id}] {spk:<20} {duration_s:6.1f}s ({duration_s/60:4.1f}m)  {word_count:5d} words  ({size_kb} KB)")

    # Write manifests
    with open(manifest_json_path, 'w') as f:
        json.dump(manifest_entries, f, indent=2)

    with open(manifest_jsonl_path, 'w') as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + '\n')

    total_dur = sum(e['duration_s'] for e in manifest_entries)
    total_words = sum(e['word_count'] for e in manifest_entries)
    print(f"\nSuccessfully prepared {len(manifest_entries)} long-form samples in: {out_dir}")
    print(f"Total audio: {total_dur:.1f}s ({total_dur/60:.1f}m) | Total words: {total_words}")
    print(f"Manifests written: {manifest_json_path} and {manifest_jsonl_path}")


if __name__ == '__main__':
    main()
