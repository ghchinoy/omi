"""
Benchmark Suite 02: Deepgram vs Modulate — Streaming transcription.

Uses LibriSpeech test-clean or custom manifest samples.
Streams at real-time pace (3200 bytes/100ms).

Usage:
    cd backend && python scripts/stt/o_benchmark_02_streaming.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / '.env')

from jiwer import wer as compute_wer
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.stt.streaming import process_audio_dg, process_audio_modulate

PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)


def normalize_for_wer(text: str) -> str:
    return PUNCT_RE.sub('', text).lower().strip()


def count_punctuation(text: str) -> Dict[str, Any]:
    marks = re.findall(r'[^\w\s]', text)
    return {'total': len(marks), 'detail': dict(sorted(((m, marks.count(m)) for m in set(marks)), key=lambda x: -x[1]))}


DEFAULT_AUDIO_DIRS = [
    Path('/tmp/stt_benchmark_audio_02'),
    Path(__file__).resolve().parents[3] / 'benchmarks' / 'data' / 'stt_benchmark_audio_02',
]
DEFAULT_RESULTS_DIRS = [
    Path('/tmp/stt_benchmark_results'),
    Path(__file__).resolve().parents[3] / 'benchmarks' / 'results',
]

CHUNK_SIZE = 3200
CHUNK_INTERVAL = 0.1


def resolve_audio_dir(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if (p / 'manifest.json').exists():
            return p
    for p in DEFAULT_AUDIO_DIRS:
        if (p / 'manifest.json').exists():
            return p
    raise FileNotFoundError('Could not find benchmark audio directory with manifest.json.')


def load_manifest(audio_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = audio_dir / 'manifest.json'
    if not manifest_path.exists():
        print('ERROR: Samples not prepared. Run first:')
        print('  python scripts/stt/n_benchmark_02_prerecorded.py --prepare')
        sys.exit(1)
    with open(manifest_path) as f:
        return cast(List[Dict[str, Any]], json.load(f))


def read_pcm_from_wav(wav_path: Path) -> bytes:
    data = wav_path.read_bytes()
    if data[:4] == b'RIFF':
        return data[44:]
    return data


async def stream_to_provider(
    audio_pcm: bytes,
    language: str,
    provider: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_interval: float = CHUNK_INTERVAL,
) -> Dict[str, Any]:
    """Stream audio and receive transcripts in parallel, tracking per-segment timing."""
    segment_log: List[Dict[str, Any]] = []
    segments_received: List[Dict[str, Any]] = []
    stream_start: List[Optional[float]] = [None]

    def stream_transcript(segments: List[Dict[str, Any]]) -> None:
        now = time.monotonic()
        for seg in segments:
            segments_received.append(seg)
            start = stream_start[0]
            elapsed = (now - start) if start else 0
            segment_log.append(
                {
                    'wall_s': round(elapsed, 3),
                    'seg_num': len(segments_received),
                    'text': seg.get('text', '')[:60],
                }
            )

    connect_start = time.monotonic()
    socket: Any = None
    try:
        if provider == 'deepgram':
            socket = await asyncio.wait_for(
                process_audio_dg(stream_transcript, language, 16000, 1, model='nova-3'),
                timeout=15,
            )
        else:
            socket = await asyncio.wait_for(
                process_audio_modulate(stream_transcript, 16000, language),
                timeout=15,
            )
    except Exception as e:
        return {'error': str(e), 'connect_time': -1}

    if socket is None:
        return {'error': 'Failed to obtain socket', 'connect_time': -1}

    connect_time = time.monotonic() - connect_start
    stream_start_val = time.monotonic()
    stream_start[0] = stream_start_val

    audio_duration_s = len(audio_pcm) / (16000 * 2)
    segs_before_finish = 0

    offset = 0
    while offset < len(audio_pcm):
        chunk = audio_pcm[offset : offset + chunk_size]
        socket.send(chunk)
        offset += chunk_size
        if chunk_interval > 0:
            await asyncio.sleep(chunk_interval)

    segs_before_finish = len(segments_received)
    audio_sent_time = time.monotonic() - stream_start_val

    if provider == 'modulate':
        try:
            await asyncio.wait_for(socket.drain_and_close(), timeout=10)
        except (asyncio.TimeoutError, Exception):
            pass
    else:
        socket.finish()
        await asyncio.sleep(3)

    total_time = time.monotonic() - stream_start_val
    first_seg_latency = segment_log[0]['wall_s'] if segment_log else -1
    segs_after_finish = len(segments_received) - segs_before_finish

    text = ' '.join(s.get('text', '') for s in segments_received).strip()

    return {
        'connect_time': connect_time,
        'first_segment_latency': first_seg_latency,
        'total_time': total_time,
        'audio_duration_s': round(audio_duration_s, 2),
        'audio_sent_time': round(audio_sent_time, 2),
        'segments_raw': len(segments_received),
        'segments_deduped': len(segments_received),
        'segs_during_stream': segs_before_finish,
        'segs_after_finish': segs_after_finish,
        'text': text,
        'words': len(text.split()) if text else 0,
        'segment_log': segment_log,
    }


async def run_benchmark(
    audio_dir_path: Optional[str] = None,
    results_dir_path: Optional[str] = None,
    tier: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_interval: float = CHUNK_INTERVAL,
) -> None:
    audio_dir = resolve_audio_dir(audio_dir_path)
    results_dir = Path(results_dir_path) if results_dir_path else DEFAULT_RESULTS_DIRS[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    dg_key = os.getenv('DEEPGRAM_API_KEY')
    mod_key = os.getenv('MODULATE_API_KEY')
    if not dg_key:
        print('ERROR: DEEPGRAM_API_KEY not set')
        sys.exit(1)
    if not mod_key:
        print('ERROR: MODULATE_API_KEY not set')
        sys.exit(1)

    manifest = load_manifest(audio_dir)
    if tier and tier.lower() != 'all':
        manifest = [c for c in manifest if c.get('tier', '').lower() == tier.lower()]

    print(f'\nBenchmark Suite 02 — Streaming ({len(manifest)} samples)')
    print(f'Audio Directory: {audio_dir}')
    print(f'Chunk pacing: {chunk_size} bytes / {chunk_interval*1000:.0f}ms\n')

    results: List[Dict[str, Any]] = []
    for case in manifest:
        wav_path = audio_dir / f"{case['id']}.wav"
        audio_pcm = read_pcm_from_wav(wav_path)
        ref_norm = normalize_for_wer(case['text'])
        lang = 'en'

        row: Dict[str, Any] = {
            'id': case['id'],
            'uid': case['uid'],
            'description': case['description'],
            'speaker': case['speaker'],
            'ref_words': case['word_count'],
            'duration_s': case['duration_s'],
            'audio_kb': len(audio_pcm) / 1024,
            'ref_text': case['text'],
        }

        print(f"  [{case['id']}] {case['description']} (speaker {case['speaker']})")

        for provider, prefix in [('deepgram', 'dg'), ('modulate', 'mod')]:
            try:
                result = await asyncio.wait_for(
                    stream_to_provider(
                        audio_pcm=audio_pcm,
                        language=lang,
                        provider=provider,
                        chunk_size=chunk_size,
                        chunk_interval=chunk_interval,
                    ),
                    timeout=200.0,
                )
                if 'error' in result:
                    raise RuntimeError(result['error'])
                wer_val: float = compute_wer(ref_norm, normalize_for_wer(result['text'])) if result['text'] else 1.0
                punct: Dict[str, Any] = (
                    count_punctuation(result['text']) if result['text'] else {'total': 0, 'detail': {}}
                )
                row.update(
                    {
                        f'{prefix}_connect': result['connect_time'],
                        f'{prefix}_first_seg': result['first_segment_latency'],
                        f'{prefix}_total': result['total_time'],
                        f'{prefix}_segments_raw': result['segments_raw'],
                        f'{prefix}_segments_deduped': result['segments_deduped'],
                        f'{prefix}_segs_during': result['segs_during_stream'],
                        f'{prefix}_segs_after': result['segs_after_finish'],
                        f'{prefix}_words': result['words'],
                        f'{prefix}_wer': wer_val,
                        f'{prefix}_text': result['text'],
                        f'{prefix}_punct': punct['total'],
                        f'{prefix}_punct_detail': punct['detail'],
                        f'{prefix}_segment_log': result['segment_log'],
                    }
                )
                during = result['segs_during_stream']
                total_raw = result['segments_raw']
                total_deduped = result['segments_deduped']
                parallel_pct = (during / total_raw * 100) if total_raw > 0 else 0
                print(
                    f"    {provider.capitalize():10s}  connect={result['connect_time']:.2f}s  "
                    f"1st_seg={result['first_segment_latency']:.2f}s  "
                    f"total={result['total_time']:.2f}s  "
                    f"segs={total_raw} (during={during} parallel={parallel_pct:.0f}% deduped={total_deduped})  "
                    f"WER={wer_val:.2%}"
                )
            except Exception as e:
                print(f"    {provider.capitalize():10s}  ERROR - {e}")
                row.update(
                    {
                        f'{prefix}_connect': -1,
                        f'{prefix}_first_seg': -1,
                        f'{prefix}_total': -1,
                        f'{prefix}_segments': 0,
                        f'{prefix}_segs_during': 0,
                        f'{prefix}_segs_after': 0,
                        f'{prefix}_words': 0,
                        f'{prefix}_wer': 1.0,
                        f'{prefix}_text': f'ERROR: {e}',
                        f'{prefix}_punct': 0,
                        f'{prefix}_punct_detail': {},
                    }
                )

        results.append(row)
        # Save results incrementally after each sample
        output_path = results_dir / 'suite02_streaming_benchmark.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    print('\n' + '=' * 130)
    print('SUITE 02 — STREAMING BENCHMARK RESULTS')
    print('=' * 130)

    output_path = results_dir / 'suite02_streaming_benchmark.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nDetailed results saved to: {output_path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Deepgram & Modulate Streaming Benchmark')
    parser.add_argument('--audio-dir', type=str, default=None, help='Path to audio directory with manifest.json')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to save JSON benchmark results')
    parser.add_argument('--tier', type=str, default='a', choices=['all', 'a', 'b', 'A', 'B'], help='Filter by tier (default: a for streaming)')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Audio chunk size in bytes (default: 3200)')
    parser.add_argument('--pace', type=float, default=2.0, help='Playback pacing (1.0 = realtime, 2.0 = 2x, 0.0 = burst)')
    args = parser.parse_args()

    interval = (args.chunk_size / 32000.0) / args.pace if args.pace > 0 else 0.0

    asyncio.run(
        run_benchmark(
            audio_dir_path=args.audio_dir,
            results_dir_path=args.results_dir,
            tier=args.tier,
            chunk_size=args.chunk_size,
            chunk_interval=interval,
        )
    )


if __name__ == '__main__':
    main()
