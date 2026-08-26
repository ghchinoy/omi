"""
Benchmark Suite: Gemini 3.5 Transcribe — Pre-recorded / Batch transcription.

Evaluates Gemini 3.5 Transcribe (`gemini-3.5-transcribe-preview`) on LibriSpeech
test-clean samples from n_benchmark_02_prerecorded.py, measuring WER (0-100% scale)
and processing latency / xRealtime throughput.

Setup:
    1. Prepare samples:
       python scripts/stt/n_benchmark_02_prerecorded.py --prepare
    2. Set credentials:
       export PROJECT_ID=your-gcp-project-id  # or GEMINI_API_KEY=...

Usage:
    cd backend && python scripts/stt/aa_benchmark_gemini_prerecorded.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / '.env')

from jiwer import wer as compute_wer
from tabulate import tabulate
from google import genai
from google.genai import types

PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)

DEFAULT_AUDIO_DIRS = [
    Path('/tmp/stt_benchmark_audio_02'),
    Path(__file__).resolve().parents[3] / 'benchmarks' / 'data' / 'stt_benchmark_audio_02',
    Path('/tmp/librispeech_benchmark_02'),
]

DEFAULT_RESULTS_DIRS = [
    Path('/tmp/stt_benchmark_results'),
    Path(__file__).resolve().parents[3] / 'benchmarks' / 'results',
]


def normalize_for_wer(text: str) -> str:
    return PUNCT_RE.sub('', text).lower().strip()


def count_punctuation(text: str) -> Dict[str, Any]:
    marks = re.findall(r'[^\w\s]', text)
    return {'total': len(marks), 'detail': dict(sorted(((m, marks.count(m)) for m in set(marks)), key=lambda x: -x[1]))}


def resolve_audio_dir(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if (p / 'manifest.json').exists():
            return p
    for p in DEFAULT_AUDIO_DIRS:
        if (p / 'manifest.json').exists():
            return p
    raise FileNotFoundError(
        'Could not find benchmark audio directory with manifest.json. Run n_benchmark_02_prerecorded.py --prepare first.'
    )


def load_manifest(audio_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = audio_dir / 'manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return cast(List[Dict[str, Any]], json.load(f))


def get_genai_client(project: Optional[str] = None, location: str = 'global', api_key: Optional[str] = None) -> genai.Client:
    api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    project = project or os.getenv('PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT')

    if project:
        return genai.Client(vertexai=True, project=project, location=location)
    elif api_key:
        return genai.Client(api_key=api_key)
    else:
        raise ValueError('Either PROJECT_ID (for Vertex AI) or GEMINI_API_KEY must be provided.')


def transcribe_gemini_sync(client: genai.Client, model: str, wav_path: Path) -> Tuple[str, float]:
    audio_bytes = wav_path.read_bytes()
    start_time = time.monotonic()

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=audio_bytes, mime_type='audio/wav')
        ],
    )
    elapsed = time.monotonic() - start_time
    transcript = resp.text.strip() if resp.text else ''
    return transcript, elapsed


def run_benchmark(
    model: str,
    audio_dir_path: Optional[str] = None,
    out_dir_path: Optional[str] = None,
    project: Optional[str] = None,
    location: str = 'global',
    api_key: Optional[str] = None,
    tier: Optional[str] = None,
) -> List[Dict[str, Any]]:
    audio_dir = resolve_audio_dir(audio_dir_path)
    manifest = load_manifest(audio_dir)
    if tier and tier.lower() != 'all':
        manifest = [c for c in manifest if c.get('tier', '').lower() == tier.lower()]
    client = get_genai_client(project=project, location=location, api_key=api_key)

    for out_p in DEFAULT_RESULTS_DIRS:
        out_p.mkdir(parents=True, exist_ok=True)
    if out_dir_path:
        Path(out_dir_path).mkdir(parents=True, exist_ok=True)

    print(f'\n' + '=' * 80)
    print(f'Benchmark: Gemini 3.5 Transcribe Sync [{model}] — Pre-recorded ({len(manifest)} samples)')
    print(f'Audio Directory: {audio_dir}')
    print(f'Source: LibriSpeech test-clean (CC BY 4.0)')
    print('=' * 80 + '\n')

    results: List[Dict[str, Any]] = []

    for case in manifest:
        wav_path = audio_dir / f"{case['id']}.wav"
        ref_text = case['text']
        ref_norm = normalize_for_wer(ref_text)
        duration_s = case['duration_s']

        row: Dict[str, Any] = {
            'id': case['id'],
            'uid': case['uid'],
            'description': case['description'],
            'speaker': case['speaker'],
            'ref_words': case['word_count'],
            'duration_s': duration_s,
            'ref_text': ref_text,
            'model': model,
            'engine': 'gemini-3.5-sync',
        }

        print(f"  [{case['id']}] {case['description']} (speaker {case['speaker']}, {duration_s:.1f}s)...", end=' ', flush=True)

        try:
            transcript, elapsed = transcribe_gemini_sync(client, model, wav_path)
            hyp_norm = normalize_for_wer(transcript)
            wer_val = compute_wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else (0.0 if not ref_norm and not hyp_norm else 1.0)
            x_rt = duration_s / elapsed if elapsed > 0 else 0.0
            punct_info = count_punctuation(transcript)

            row.update({
                'transcript': transcript,
                'wer': round(wer_val * 100, 2),
                'latency_s': round(elapsed, 3),
                'x_realtime': round(x_rt, 2),
                'punct_count': punct_info['total'],
                'hyp_words': len(transcript.split()),
                'hyp_chars': len(transcript),
            })
            print(f"WER={wer_val*100:.1f}% | lat={elapsed:.2f}s | {x_rt:.2f}x speed")
        except Exception as e:
            print(f"ERROR: {e}")
            row.update({
                'transcript': f'ERROR: {e}',
                'wer': None,
                'latency_s': None,
                'x_realtime': None,
                'error': str(e),
            })

        results.append(row)

    # Save results to output destinations
    safe_model_name = model.replace('/', '_').replace(':', '_')
    json_filename = f'gemini_prerecorded_results_{safe_model_name}.json'
    
    save_dirs = list(DEFAULT_RESULTS_DIRS)
    if out_dir_path:
        save_dirs.append(Path(out_dir_path))

    for d in save_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / json_filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save to {d}: {e}")

    # Summary table
    print('\n' + '=' * 80)
    print(f'RESULTS SUMMARY: Gemini 3.5 Transcribe [{model}]')
    print('=' * 80)

    table: List[List[Any]] = []
    wers: List[float] = []
    lats: List[float] = []
    xrts: List[float] = []
    total_audio = sum(r['duration_s'] for r in results)
    total_wall = sum(r['latency_s'] for r in results if r.get('latency_s') is not None)

    for r in results:
        w = r.get('wer')
        l = r.get('latency_s')
        x = r.get('x_realtime')
        if w is not None:
            wers.append(w)
        if l is not None:
            lats.append(l)
        if x is not None:
            xrts.append(x)

        table.append([
            r['id'],
            f"{r['duration_s']:.1f}s",
            r['ref_words'],
            f"{w:.1f}%" if w is not None else 'ERR',
            f"{l:.2f}s" if l is not None else 'ERR',
            f"{x:.2f}x" if x is not None else 'ERR',
            r.get('transcript', '')[:40] + ('...' if len(r.get('transcript', '')) > 40 else '')
        ])

    print(tabulate(table, headers=['Sample', 'Duration', 'Words', 'WER (%)', 'Latency', 'Speed', 'Transcript Preview']))

    mean_wer = sum(wers) / len(wers) if wers else 0.0
    mean_lat = sum(lats) / len(lats) if lats else 0.0
    mean_xrt = total_audio / total_wall if total_wall > 0 else 0.0

    print(f"\n  • Aggregate WER:     {mean_wer:.2f}%")
    print(f"  • Mean Latency:      {mean_lat:.2f}s")
    print(f"  • Overall Speed:     {mean_xrt:.2f}x realtime ({total_audio:.1f}s audio / {total_wall:.2f}s wall)")
    print(f"  • Success Rate:      {len(wers)}/{len(results)} clips ({len(wers)*100/len(results):.0f}%)")
    print(f"  • Results saved to:  {DEFAULT_RESULTS_DIRS[0] / json_filename}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Gemini 3.5 Transcribe Pre-recorded / Batch ASR Benchmark')
    parser.add_argument('--model', type=str, default='gemini-3.5-transcribe-preview', help='Gemini model identifier')
    parser.add_argument('--audio-dir', type=str, default=None, help='Path to audio directory with manifest.json')
    parser.add_argument('--out-dir', type=str, default=None, help='Directory to save JSON benchmark results')
    parser.add_argument('--tier', type=str, default='all', choices=['all', 'a', 'b', 'A', 'B'], help='Filter by tier (a, b, or all)')
    parser.add_argument('--project', type=str, default=None, help='GCP Project ID for Vertex AI')
    parser.add_argument('--location', type=str, default='global', help='Vertex AI location (default: global)')
    parser.add_argument('--api-key', type=str, default=None, help='Gemini Developer API Key')
    args = parser.parse_args()

    run_benchmark(
        model=args.model,
        audio_dir_path=args.audio_dir,
        out_dir_path=args.out_dir,
        project=args.project,
        location=args.location,
        api_key=args.api_key,
        tier=args.tier,
    )


if __name__ == '__main__':
    main()
