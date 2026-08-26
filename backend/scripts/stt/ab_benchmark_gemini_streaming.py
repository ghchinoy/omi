"""
Benchmark Suite: Gemini 3.5 Transcribe — Real-Time Streaming transcription.

Evaluates Gemini 3.5 Transcribe Live (`gemini-3.5-transcribe-live-preview`) streaming
performance using LibriSpeech test-clean samples. Streams audio at real-time pace
(3200 bytes / 100ms) matching Omi wearable device microphone output format.

Measures:
    - Connection latency (handshake time)
    - Time-to-First-Transcript (TTFT)
    - Total stream time and Realtime Factor
    - Intermediate / Final segment count
    - Streaming Word Error Rate (WER)

Setup:
    1. Prepare samples:
       python scripts/stt/n_benchmark_02_prerecorded.py --prepare
    2. Set credentials:
       export PROJECT_ID=your-gcp-project-id  # or GEMINI_API_KEY=...

Usage:
    cd backend && python scripts/stt/ab_benchmark_gemini_streaming.py
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

CHUNK_SIZE = 3200       # 3200 bytes @ 16kHz 16-bit mono = 100ms
CHUNK_INTERVAL = 0.1    # 100ms real-time chunk interval


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
    raise FileNotFoundError(
        'Could not find benchmark audio directory with manifest.json. Run n_benchmark_02_prerecorded.py --prepare first.'
    )


def load_manifest(audio_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = audio_dir / 'manifest.json'
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return cast(List[Dict[str, Any]], json.load(f))


def read_pcm_from_wav(wav_path: Path) -> bytes:
    with _wave.open(str(wav_path), 'rb') as wf:
        return wf.readframes(wf.getnframes())


def get_genai_client(project: Optional[str] = None, location: str = 'global', api_key: Optional[str] = None) -> genai.Client:
    project = project or os.getenv('PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT')
    api_key = api_key or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

    if project:
        return genai.Client(vertexai=True, project=project, location=location)
    elif api_key:
        return genai.Client(api_key=api_key)
    else:
        # Default to Vertex AI with ADC
        return genai.Client(vertexai=True, project='generative-bazaar-001', location=location)


class TranscriptCollector:
    """Collects streaming interim and finalized transcription events, avoiding duplication."""
    def __init__(self):
        self.finals: List[str] = []
        self.committed: List[str] = []
        self.interims_count = 0
        self.last_interim: str = ""

    def add(self, text: str, is_final: bool) -> None:
        text = text.strip()
        if not text:
            return
        if is_final:
            self.finals.append(text)
            return

        self.interims_count += 1
        if self.last_interim:
            prev_tokens = self.last_interim.lower().split()
            curr_tokens = text.lower().split()
            # If current interim doesn't continue previous interim, commit previous
            if len(curr_tokens) < len(prev_tokens) or curr_tokens[:min(len(prev_tokens), 2)] != prev_tokens[:min(len(prev_tokens), 2)]:
                if len(prev_tokens) > 1 or (prev_tokens and prev_tokens[0] in {'.', '!', '?'}):
                    self.committed.append(self.last_interim)
        self.last_interim = text

    def get_transcript(self) -> str:
        if self.finals:
            return ' '.join(self.finals).strip()
        all_chunks = list(self.committed)
        if self.last_interim and (not all_chunks or all_chunks[-1] != self.last_interim):
            all_chunks.append(self.last_interim)
        return ' '.join(all_chunks).strip()


async def stream_to_gemini(
    client: genai.Client,
    model: str,
    audio_pcm: bytes,
    chunk_size: int = CHUNK_SIZE,
    chunk_interval: float = CHUNK_INTERVAL,
    language_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Streams 16kHz PCM audio to Gemini Live session and collects interim & finalized segments."""
    col = TranscriptCollector()
    first_segment_time: List[Optional[float]] = [None]
    total_segments = 0

    transcription_config = types.AudioTranscriptionConfig()
    if language_codes:
        transcription_config.language_hints = types.LanguageHints(language_codes=language_codes)
    else:
        transcription_config.language_auto = types.LanguageAuto()

    live_config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        input_audio_transcription=transcription_config,
    )

    connect_start = time.monotonic()
    async with client.aio.live.connect(model=model, config=live_config) as session:
        connect_time = time.monotonic() - connect_start
        stream_start = time.monotonic()

        async def sender():
            for i in range(0, len(audio_pcm), chunk_size):
                chunk = audio_pcm[i : i + chunk_size]
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type='audio/pcm;rate=16000')
                )
                if chunk_interval > 0:
                    await asyncio.sleep(chunk_interval)
            await session.send_realtime_input(audio_stream_end=True)

        async def receiver():
            nonlocal total_segments
            try:
                async for message in session.receive():
                    now = time.monotonic()
                    if message.server_content:
                        sc = message.server_content
                        if getattr(sc, 'interim_input_transcription', None):
                            t = sc.interim_input_transcription.text
                            if t:
                                if first_segment_time[0] is None:
                                    first_segment_time[0] = now
                                col.add(t, False)
                                total_segments += 1

                        if getattr(sc, 'input_transcription', None):
                            t = sc.input_transcription.text
                            if t:
                                if first_segment_time[0] is None:
                                    first_segment_time[0] = now
                                col.add(t, True)
                                total_segments += 1

                        if getattr(sc, 'turn_complete', False):
                            break
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Catch API error gracefully (e.g. queue overrun or session closure)
                pass

        sender_task = asyncio.create_task(sender())
        receiver_task = asyncio.create_task(receiver())

        await sender_task
        try:
            await asyncio.wait_for(receiver_task, timeout=4.0)
        except asyncio.TimeoutError:
            pass

        total_time = time.monotonic() - stream_start

    transcript = col.get_transcript()
    ttft = (first_segment_time[0] - stream_start) if first_segment_time[0] is not None else None

    return {
        'transcript': transcript,
        'finals_count': len(col.finals),
        'interims_count': col.interims_count,
        'connect_time': round(connect_time, 3),
        'first_segment_s': round(ttft, 3) if ttft is not None else None,
        'total_time': round(total_time, 3),
        'segments': total_segments,
    }


async def run_streaming_benchmark(
    model: str,
    audio_dir_path: Optional[str] = None,
    out_dir_path: Optional[str] = None,
    project: Optional[str] = None,
    location: str = 'global',
    api_key: Optional[str] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_interval: float = CHUNK_INTERVAL,
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
    print(f'Benchmark: Gemini 3.5 Transcribe Live [{model}] — Real-Time Streaming ({len(manifest)} samples)')
    print(f'Audio Directory: {audio_dir}')
    print(f'Chunk Pacing: {chunk_size} bytes / {chunk_interval*1000:.0f}ms (Omi 16kHz mono format)')
    print('=' * 80 + '\n')

    results: List[Dict[str, Any]] = []

    for case in manifest:
        wav_path = audio_dir / f"{case['id']}.wav"
        audio_pcm = read_pcm_from_wav(wav_path)
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
            'engine': 'gemini-3.5-live',
        }

        print(f"  [{case['id']}] {case['description']} ({duration_s:.1f}s)...", end=' ', flush=True)

        try:
            stream_res = await stream_to_gemini(
                client=client,
                model=model,
                audio_pcm=audio_pcm,
                chunk_size=chunk_size,
                chunk_interval=chunk_interval,
            )
            transcript = stream_res['transcript']
            hyp_norm = normalize_for_wer(transcript)
            wer_val = compute_wer(ref_norm, hyp_norm) if ref_norm and hyp_norm else (0.0 if not ref_norm and not hyp_norm else 1.0)
            x_rt = duration_s / stream_res['total_time'] if stream_res['total_time'] > 0 else 0.0

            row.update({
                'transcript': transcript,
                'wer': round(wer_val * 100, 2),
                'connect_time_s': stream_res['connect_time'],
                'ttft_s': stream_res['first_segment_s'],
                'total_time_s': stream_res['total_time'],
                'x_realtime': round(x_rt, 2),
                'segments': stream_res['segments'],
                'hyp_words': len(transcript.split()),
                'hyp_chars': len(transcript),
            })
            ttft_str = f"{stream_res['first_segment_s']:.2f}s" if stream_res['first_segment_s'] else 'N/A'
            print(f"WER={wer_val*100:.1f}% | TTFT={ttft_str} | conn={stream_res['connect_time']:.2f}s | segs={stream_res['segments']}")
        except Exception as e:
            print(f"ERROR: {e}")
            row.update({
                'transcript': f'ERROR: {e}',
                'wer': None,
                'connect_time_s': None,
                'ttft_s': None,
                'total_time_s': None,
                'x_realtime': None,
                'error': str(e),
            })

        results.append(row)

    # Save results to output destinations
    safe_model_name = model.replace('/', '_').replace(':', '_')
    json_filename = f'gemini_streaming_results_{safe_model_name}.json'

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
    print(f'STREAMING RESULTS SUMMARY: Gemini 3.5 Transcribe Live [{model}]')
    print('=' * 80)

    table: List[List[Any]] = []
    wers: List[float] = []
    ttfts: List[float] = []
    conns: List[float] = []

    for r in results:
        w = r.get('wer')
        ttft = r.get('ttft_s')
        conn = r.get('connect_time_s')
        if w is not None:
            wers.append(w)
        if ttft is not None:
            ttfts.append(ttft)
        if conn is not None:
            conns.append(conn)

        table.append([
            r['id'],
            f"{r['duration_s']:.1f}s",
            r['ref_words'],
            f"{w:.1f}%" if w is not None else 'ERR',
            f"{conn:.2f}s" if conn is not None else 'ERR',
            f"{ttft:.2f}s" if ttft is not None else 'ERR',
            r.get('segments', 0),
            r.get('transcript', '')[:40] + ('...' if len(r.get('transcript', '')) > 40 else '')
        ])

    print(tabulate(table, headers=['Sample', 'Duration', 'Words', 'WER (%)', 'Connect', 'TTFT', 'Segs', 'Transcript Preview']))

    mean_wer = sum(wers) / len(wers) if wers else 0.0
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    mean_conn = sum(conns) / len(conns) if conns else 0.0

    print(f"\n  • Aggregate Streaming WER:  {mean_wer:.2f}%")
    print(f"  • Mean Connect Time:        {mean_conn:.2f}s")
    print(f"  • Mean Time-to-First (TTFT): {mean_ttft:.2f}s")
    print(f"  • Success Rate:             {len(wers)}/{len(results)} clips ({len(wers)*100/len(results):.0f}%)")
    print(f"  • Results saved to:         {DEFAULT_RESULTS_DIRS[0] / json_filename}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Gemini 3.5 Transcribe Live Streaming ASR Benchmark')
    parser.add_argument('--model', type=str, default='gemini-3.5-transcribe-live-preview', help='Gemini Live model identifier')
    parser.add_argument('--audio-dir', type=str, default=None, help='Path to audio directory with manifest.json')
    parser.add_argument('--out-dir', type=str, default=None, help='Directory to save JSON benchmark results')
    parser.add_argument('--tier', type=str, default='a', choices=['all', 'a', 'b', 'A', 'B'], help='Filter by tier (default: a for streaming)')
    parser.add_argument('--project', type=str, default=None, help='GCP Project ID for Vertex AI')
    parser.add_argument('--location', type=str, default='global', help='Vertex AI location (default: global)')
    parser.add_argument('--api-key', type=str, default=None, help='Gemini Developer API Key')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help='Audio chunk size in bytes (default: 3200)')
    parser.add_argument('--pace', type=float, default=2.0, help='Playback pacing (1.0 = realtime, 2.0 = 2x, 0.0 = burst)')
    args = parser.parse_args()

    interval = (args.chunk_size / 32000.0) / args.pace if args.pace > 0 else 0.0

    asyncio.run(
        run_streaming_benchmark(
            model=args.model,
            audio_dir_path=args.audio_dir,
            out_dir_path=args.out_dir,
            project=args.project,
            location=args.location,
            api_key=args.api_key,
            chunk_size=args.chunk_size,
            chunk_interval=interval,
            tier=args.tier,
        )
    )


if __name__ == '__main__':
    main()
