"""
Benchmark Suite: Moonshine STT — Pre-recorded / Batch transcription.

Evaluates Moonshine STT on LibriSpeech or custom manifest samples, measuring
WER (0-100% scale) and processing latency.

Setup:
    1. Start moonshine serve:
       moonshine serve --transport ws --addr :8768 --audio-source remote --remote-audio-encoding int16 --remote-audio-rate 16000
    2. Set env var:
       HOSTED_MOONSHINE_WS_URL=ws://localhost:8768/ws

Usage:
    cd backend && python scripts/stt/y_benchmark_moonshine_prerecorded.py
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
from typing import Any, Dict, List, Optional, Tuple, cast

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
    raise FileNotFoundError('Could not find benchmark audio directory with manifest.json.')


def load_manifest(audio_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = audio_dir / 'manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return cast(List[Dict[str, Any]], json.load(f))


def read_pcm_from_wav(wav_path: Path) -> bytes:
    with _wave.open(str(wav_path), 'rb') as wf:
        return wf.readframes(wf.getnframes())


async def transcribe_moonshine(audio_pcm: bytes, ws_url: str = 'ws://localhost:8768/ws') -> Tuple[str, float, int]:
    t0 = time.monotonic()
    async with websockets.connect(ws_url) as ws:
        finalized_texts: List[str] = []
        seen_ids = set()
        latest_lines: List[Dict[str, Any]] = []
        done = asyncio.Event()

        async def send_audio():
            chunk_size = 32000  # 1-second chunks
            for i in range(0, len(audio_pcm), chunk_size):
                await ws.send(audio_pcm[i : i + chunk_size])
                await asyncio.sleep(0.005)
            # Wait for Moonshine to finalize remaining audio
            await asyncio.sleep(1.5)
            done.set()

        async def recv_transcripts():
            while not done.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
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
                                        finalized_texts.append(txt)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

        sender = asyncio.create_task(send_audio())
        receiver = asyncio.create_task(recv_transcripts())
        await sender
        await receiver

        # Flush unfinalized trailing lines
        for l in latest_lines:
            if l.get('id') not in seen_ids and l.get('text', '').strip():
                finalized_texts.append(l['text'].strip())

        elapsed = time.monotonic() - t0
        text = ' '.join(finalized_texts).strip()
        return text, elapsed, len(finalized_texts)


async def async_main(
    arch: str,
    audio_dir_path: Optional[str] = None,
    results_dir_path: Optional[str] = None,
    tier: Optional[str] = None,
) -> None:
    audio_dir = resolve_audio_dir(audio_dir_path)
    manifest = load_manifest(audio_dir)
    if tier and tier.lower() != 'all':
        manifest = [c for c in manifest if c.get('tier', '').lower() == tier.lower()]

    results_dir = Path(results_dir_path) if results_dir_path else DEFAULT_RESULTS_DIRS[0]
    results_dir.mkdir(parents=True, exist_ok=True)

    ws_url = os.getenv('HOSTED_MOONSHINE_WS_URL', 'ws://localhost:8768/ws')

    print(f'\nBenchmark: Moonshine STT [{arch}] — Pre-recorded ({len(manifest)} samples)')
    print(f'Audio Directory: {audio_dir}')
    print(f'Moonshine endpoint: {ws_url}\n')

    results: List[Dict[str, Any]] = []
    for case in manifest:
        wav_path = audio_dir / f"{case['id']}.wav"
        audio_pcm = read_pcm_from_wav(wav_path)
        ref_norm = normalize_for_wer(case['text'])

        row: Dict[str, Any] = {
            'id': case['id'],
            'uid': case['uid'],
            'description': case['description'],
            'speaker': case['speaker'],
            'ref_words': case['word_count'],
            'duration_s': case['duration_s'],
            'ref_text': case['text'],
            'ms_model': arch,
        }

        print(f"  [{case['id']}] {case['description']} (speaker {case['speaker']})")

        try:
            ms_text, ms_time, ms_segments = await transcribe_moonshine(audio_pcm, ws_url)
            ms_norm = normalize_for_wer(ms_text)
            ms_wer = compute_wer(ref_norm, ms_norm) if ref_norm and ms_norm else (0.0 if not ref_norm and not ms_norm else 1.0)
            ms_punct = count_punctuation(ms_text)
            row.update(
                {
                    'ms_text': ms_text,
                    'ms_wer': round(ms_wer * 100, 1),
                    'ms_latency_s': round(ms_time, 2),
                    'ms_segments': ms_segments,
                    'ms_punct': ms_punct['total'],
                }
            )
            print(f"    Moonshine [{arch}]: WER={ms_wer*100:.1f}% lat={ms_time:.2f}s segs={ms_segments}")
        except Exception as e:
            print(f"    Moonshine ERROR: {e}")
            row.update({'ms_text': f'ERROR: {e}', 'ms_wer': None, 'ms_latency_s': None})

        results.append(row)

    out_file = results_dir / f'moonshine_prerecorded_results_{arch}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 80)
    print(f'RESULTS SUMMARY [{arch}]')
    print('=' * 80)

    table: List[List[Any]] = []
    ms_wers: List[float] = []
    ms_lats: List[float] = []

    for r in results:
        ms_w = r.get('ms_wer')
        ms_l = r.get('ms_latency_s')
        if ms_w is not None:
            ms_wers.append(ms_w)
        if ms_l is not None:
            ms_lats.append(ms_l)

        table.append(
            [
                r['id'],
                f"{r['duration_s']:.1f}s",
                r['ref_words'],
                f"{ms_w:.1f}%" if ms_w is not None else 'ERR',
                f"{ms_l:.2f}s" if ms_l is not None else 'ERR',
            ]
        )

    print(tabulate(table, headers=['Sample', 'Duration', 'Words', f'Moonshine [{arch}] WER', 'Latency']))

    print(f'\n  Moonshine [{arch}] avg WER:  {sum(ms_wers)/len(ms_wers):.1f}%' if ms_wers else '')
    print(f'  Moonshine [{arch}] avg lat:  {sum(ms_lats)/len(ms_lats):.2f}s' if ms_lats else '')
    print(f'\n  Results saved to: {out_file}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Moonshine STT Pre-recorded ASR Benchmark')
    parser.add_argument('--arch', type=str, default='tiny-streaming', help='Moonshine model architecture token (e.g. tiny-streaming, small-streaming, base)')
    parser.add_argument('--audio-dir', type=str, default=None, help='Path to audio directory with manifest.json')
    parser.add_argument('--results-dir', type=str, default=None, help='Directory to save JSON benchmark results')
    parser.add_argument('--tier', type=str, default='all', choices=['all', 'a', 'b', 'A', 'B'], help='Filter by tier (a, b, or all)')
    args = parser.parse_args()

    asyncio.run(
        async_main(
            arch=args.arch,
            audio_dir_path=args.audio_dir,
            results_dir_path=args.results_dir,
            tier=args.tier,
        )
    )


if __name__ == '__main__':
    main()
