"""
Benchmark Suite: Moonshine STT — Streaming transcription.

Evaluates Moonshine STT streaming performance using LibriSpeech or custom manifest samples.
Streams audio at real-time or accelerated pace (3200 bytes / 100ms) matching Omi wearable device format.

Setup:
    1. Start moonshine serve:
       moonshine serve --transport ws --addr :8768 --audio-source remote --remote-audio-encoding int16 --remote-audio-rate 16000
    2. Set env var:
       HOSTED_MOONSHINE_WS_URL=ws://localhost:8768/ws

Usage:
    cd backend && python scripts/stt/z_benchmark_moonshine_streaming.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import wave as _wave
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / '.env')

from jiwer import wer as compute_wer
from tabulate import tabulate
import websockets

PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)

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


def normalize_for_wer(text: str) -> str:
    return PUNCT_RE.sub('', text).lower().strip()


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
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return cast(List[Dict[str, Any]], json.load(f))


def read_pcm_from_wav(wav_path: Path) -> bytes:
    with _wave.open(str(wav_path), 'rb') as wf:
        return wf.readframes(wf.getnframes())


async def stream_to_moonshine(
    audio_pcm: bytes,
    ws_url: str = 'ws://localhost:8768/ws',
    chunk_size: int = CHUNK_SIZE,
    chunk_interval: float = CHUNK_INTERVAL,
) -> Dict[str, Any]:
    connect_start = time.monotonic()
    async with websockets.connect(ws_url) as ws:
        connect_time = time.monotonic() - connect_start
        finalized_texts: List[str] = []
        seen_ids = set()
        latest_lines: List[Dict[str, Any]] = []
        first_segment_time: List[Optional[float]] = [None]
        total_segments = 0
        done = asyncio.Event()

        stream_start = time.monotonic()

        async def send_audio():
            for i in range(0, len(audio_pcm), chunk_size):
                await ws.send(audio_pcm[i : i + chunk_size])
                if chunk_interval > 0:
                    await asyncio.sleep(chunk_interval)
            await asyncio.sleep(1.5)
            done.set()

        async def recv_transcripts():
            nonlocal total_segments
            while not done.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    now = time.monotonic()
                    data = json.loads(msg)
                    if data.get('kind') == 'transcript':
                        payload = data.get('payload', {})
                        lines = payload.get('lines') or []
                        finalized_ids = payload.get('finalized_line_ids') or []
                        lines_by_id = {l['id']: l for l in lines}
                        latest_lines.clear()
                        latest_lines.extend(lines)
                        for lid in finalized_ids:
                            if lid not in seen_ids:
                                seen_ids.add(lid)
                                if lid in lines_by_id:
                                    txt = lines_by_id[lid].get('text', '').strip()
                                    if txt:
                                        if first_segment_time[0] is None:
                                            first_segment_time[0] = now
                                        finalized_texts.append(txt)
                                        total_segments += 1
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(recv_transcripts())
        await sender
        await receiver

        # Flush unfinalized
        for l in latest_lines:
            if l.get('id') not in seen_ids and l.get('text', '').strip():
                finalized_texts.append(l['text'].strip())
                total_segments += 1

        total_time = time.monotonic() - stream_start
        text = ' '.join(finalized_texts).strip()
        first_seg = (first_segment_time[0] - stream_start) if first_segment_time[0] else None

        return {
            'text': text,
            'connect_time': round(connect_time, 3),
            'first_segment_s': round(first_seg, 3) if first_seg else None,
            'total_time': round(total_time, 3),
            'segments': total_segments,
        }


async def async_main(
    arch: str,
    audio_dir_path: Optional[str] = None,
    results_dir_path: Optional[str] = None,
    tier: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_interval: float = CHUNK_INTERVAL,
) -> None:
    audio_dir = resolve_audio_dir(audio_dir_path)
    manifest = load_manifest(audio_dir)
    if tier and tier.lower() != 'all':
        manifest = [c for c in manifest if c.get('tier', '').lower() == tier.lower()]

    results_dir = Path(results_dir_path) if results_dir_path else DEFAULT_RESULTS_DIRS[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    ws_url = os.getenv('HOSTED_MOONSHINE_WS_URL', 'ws://localhost:8768/ws')

    print(f'\nBenchmark: Moonshine STT [{arch}] — Streaming ({len(manifest)} samples)')
    print(f'Audio Directory: {audio_dir}')
    print(f'Moonshine endpoint: {ws_url}')
    print(f'Chunk pacing: {chunk_size} bytes / {chunk_interval*1000:.0f}ms\n')

    results: List[Dict[str, Any]] = []
    for case in manifest:
        wav_path = audio_dir / f"{case['id']}.wav"
        audio_pcm = read_pcm_from_wav(wav_path)
        ref_norm = normalize_for_wer(case['text'])

        row: Dict[str, Any] = {
            'id': case['id'],
            'description': case['description'],
            'duration_s': case['duration_s'],
            'ref_words': case['word_count'],
            'ref_text': case['text'],
            'ms_model': arch,
        }

        print(f"  [{case['id']}] {case['description']} ({case['duration_s']:.1f}s)...", end=' ', flush=True)

        try:
            ms = await stream_to_moonshine(
                audio_pcm=audio_pcm,
                ws_url=ws_url,
                chunk_size=chunk_size,
                chunk_interval=chunk_interval,
            )
            ms_norm = normalize_for_wer(ms['text'])
            ms_wer = compute_wer(ref_norm, ms_norm) if ref_norm and ms_norm else (0.0 if not ref_norm and not ms_norm else 1.0)
            row.update(
                {
                    'ms_text': ms['text'],
                    'ms_wer': round(ms_wer * 100, 1),
                    'ms_connect_s': ms['connect_time'],
                    'ms_first_seg_s': ms['first_segment_s'],
                    'ms_total_s': ms['total_time'],
                    'ms_segments': ms['segments'],
                }
            )
            first_s_str = f"{ms['first_segment_s']:.2f}s" if ms['first_segment_s'] else 'N/A'
            print(f"WER={ms_wer*100:.1f}% | 1st_seg={first_s_str} | conn={ms['connect_time']:.2f}s | segs={ms['segments']}")
        except Exception as e:
            print(f"ERROR: {e}")
            row.update({'ms_text': f'ERROR: {e}', 'ms_wer': None, 'ms_first_seg_s': None})

        results.append(row)

    out_file = results_dir / f'moonshine_streaming_results_{arch}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 80)
    print(f'RESULTS SUMMARY [{arch}]')
    print('=' * 80)

    table: List[List[Any]] = []
    for r in results:
        table.append(
            [
                r['id'],
                f"{r['duration_s']:.1f}s",
                f"{r.get('ms_wer', 'ERR')}%",
                f"{r.get('ms_first_seg_s', 'ERR')}s",
                r.get('ms_segments', '-'),
            ]
        )

    print(tabulate(table, headers=['Sample', 'Dur', f'MS [{arch}] WER', 'MS 1st Seg', 'MS Segs']))
    print(f'\n  Results saved to: {out_file}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Moonshine STT Streaming ASR Benchmark')
    parser.add_argument('--arch', type=str, default='tiny-streaming', help='Moonshine model architecture token (e.g. tiny-streaming, small-streaming, base)')
    parser.add_argument('--audio-dir', type=str, default=None, help='Path to audio directory with manifest.json')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to save JSON benchmark results')
    parser.add_argument('--tier', type=str, default='a', choices=['all', 'a', 'b', 'A', 'B'], help='Filter by tier (default: a for streaming)')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Audio chunk size in bytes (default: 3200)')
    parser.add_argument('--pace', type=float, default=2.0, help='Playback pacing (1.0 = realtime, 2.0 = 2x, 0.0 = burst)')
    args = parser.parse_args()

    interval = (args.chunk_size / 32000.0) / args.pace if args.pace > 0 else 0.0

    asyncio.run(
        async_main(
            arch=args.arch,
            audio_dir_path=args.audio_dir,
            results_dir_path=args.results_dir,
            tier=args.tier,
            chunk_size=args.chunk_size,
            chunk_interval=interval,
        )
    )


if __name__ == '__main__':
    main()
