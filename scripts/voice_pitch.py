from __future__ import annotations

import argparse
import csv
import logging
import math
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio

import turns as turntable
from common import setup_logging
from s3 import S3Store

logger = logging.getLogger(__name__)

HEADER_PROBE = 512
COLUMNS = [
    "wav_name", "podcast", "episode_id",
    "f0_ch0", "f0_ch1", "iqr_ch0", "iqr_ch1",
    "voiced_sec_ch0", "voiced_sec_ch1", "semitones", "effect_size", "note",
]


def wav_layout(head: bytes) -> tuple[int, int, int, int]:
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    offset, rate, channels, bits = 12, 0, 0, 0
    while offset < len(head) - 8:
        cid = head[offset : offset + 4]
        size = struct.unpack("<I", head[offset + 4 : offset + 8])[0]
        if cid == b"fmt ":
            channels, rate = struct.unpack("<HI", head[offset + 10 : offset + 16])
            bits = struct.unpack("<H", head[offset + 22 : offset + 24])[0]
        elif cid == b"data":
            return offset + 8, rate, channels, channels * bits // 8
        offset += 8 + size + (size % 2)
    raise ValueError("no data chunk in the probed header")


def clean_turns(table: turntable.TurnTable, channel: int, min_len: float):
    others = [(t.start, t.end) for t in table.turns if t.channel != channel]
    out = []
    for turn in table.turns:
        if turn.channel != channel or turn.duration < min_len:
            continue
        if any(o0 < turn.end and o1 > turn.start for o0, o1 in others):
            continue
        out.append(turn)
    return out


def pick_windows(table, channel, want_sec, window, min_len):
    candidates = clean_turns(table, channel, min_len)
    if not candidates:
        return []
    span_start = min(t.start for t in table.turns)
    span_end = max(t.end for t in table.turns)
    buckets: dict[int, list] = {}
    for turn in candidates:
        decile = int(10 * (turn.start - span_start) / max(span_end - span_start, 1e-9))
        buckets.setdefault(min(decile, 9), []).append(turn)
    ordered = []
    for decile in sorted(buckets):
        ordered.append(max(buckets[decile], key=lambda t: t.duration))
    ordered += sorted(
        (t for t in candidates if t not in ordered), key=lambda t: t.duration, reverse=True
    )
    picked, total = [], 0.0
    for turn in ordered:
        if total >= want_sec:
            break
        take = min(window, turn.duration - 0.2)
        middle = (turn.start + turn.end) / 2
        picked.append((middle - take / 2, take))
        total += take
    return picked


def channel_pitch(samples: np.ndarray, rate: int) -> tuple[float, float, float]:
    voiced = samples[samples != 0.0]
    if len(voiced) < rate:
        return math.nan, math.nan, len(voiced) / rate
    tensor = torch.from_numpy(np.ascontiguousarray(voiced))[None]
    f0 = torchaudio.functional.detect_pitch_frequency(
        tensor, rate, freq_low=60, freq_high=400
    )[0].numpy()
    f0 = f0[(f0 > 60) & (f0 < 400)]
    if len(f0) == 0:
        return math.nan, math.nan, len(voiced) / rate
    q1, q3 = np.percentile(f0, [25, 75])
    return float(np.median(f0)), float(q3 - q1), len(voiced) / rate


def measure(table, store, stereo_dir, want_sec, window, min_len, workers) -> dict:
    row = {
        "wav_name": table.wav_name, "podcast": table.podcast,
        "episode_id": table.episode_id, "note": "",
    }
    local = stereo_dir / table.wav_name if stereo_dir else None
    if local is not None and local.exists():
        head = local.open("rb").read(HEADER_PROBE)
        fetch = lambda a, b: _read_local(local, a, b)  # noqa: E731
    else:
        head = store.download_range(f"stereo/{table.wav_name}", 0, HEADER_PROBE - 1)
        fetch = lambda a, b: store.download_range(f"stereo/{table.wav_name}", a, b)  # noqa: E731
    data_offset, rate, channels, frame = wav_layout(head)
    if channels != 2:
        row["note"] = f"{channels} channel(s), expected stereo"
        return row

    # Every window is an independent range GET and the pass spends its time
    # waiting on them, not computing -- serially this runs at ~20% CPU. Fetch
    # them together and the episode cost drops by most of that idle time.
    requests = []
    for channel in (0, 1):
        for start, take in pick_windows(table, channel, want_sec, window, min_len):
            first = data_offset + max(0, int(start * rate)) * frame
            requests.append((channel, first, first + int(take * rate) * frame - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        blobs = list(pool.map(lambda r: fetch(r[1], r[2]), requests))

    stats = {}
    for channel in (0, 1):
        chunks = []
        for (chan, _, _), raw in zip(requests, blobs):
            if chan != channel:
                continue
            block = np.frombuffer(raw[: len(raw) // frame * frame], dtype="<i2")
            block = block.reshape(-1, channels).astype(np.float32) / 32768.0
            chunks.append(block[:, channel])
        samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        stats[channel] = channel_pitch(samples, rate)

    (f0_0, iqr0, v0), (f0_1, iqr1, v1) = stats[0], stats[1]
    row.update({
        "f0_ch0": round(f0_0, 1) if f0_0 == f0_0 else "",
        "f0_ch1": round(f0_1, 1) if f0_1 == f0_1 else "",
        "iqr_ch0": round(iqr0, 1) if iqr0 == iqr0 else "",
        "iqr_ch1": round(iqr1, 1) if iqr1 == iqr1 else "",
        "voiced_sec_ch0": round(v0, 1), "voiced_sec_ch1": round(v1, 1),
    })
    if f0_0 == f0_0 and f0_1 == f0_1:
        row["semitones"] = round(abs(12 * math.log2(f0_1 / f0_0)), 2)
        spread = (iqr0 + iqr1) / 2
        row["effect_size"] = round(abs(f0_1 - f0_0) / spread, 2) if spread > 0 else ""
    else:
        row["semitones"] = ""
        row["effect_size"] = ""
        row["note"] = "not enough voiced audio on one channel"
    return row


def _read_local(path: Path, first: int, last: int) -> bytes:
    with path.open("rb") as fh:
        fh.seek(first)
        return fh.read(last - first + 1)


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns-dir", type=Path, default=Path("./data/turns"))
    parser.add_argument("--stereo-dir", type=Path, default=None,
                        help="Read locally when the wavs are already here; otherwise "
                             "range-GET them from S3.")
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-prefix", default="corpus")
    parser.add_argument("--want-sec", type=float, default=24.0,
                        help="Voiced seconds to gather per channel (default: %(default)s).")
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--min-turn", type=float, default=3.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent range GETs per episode (default: %(default)s).")
    parser.add_argument("--out", type=Path, default=Path("./reports/voice_pitch.csv"))
    args = parser.parse_args()

    tables = [turntable.load(p) for p in sorted(args.turns_dir.glob("*.turns.json"))]
    if args.limit:
        tables = tables[: args.limit]
    store = S3Store(args.s3_bucket, args.s3_prefix)
    if not store.enabled and not args.stereo_dir:
        logger.error("need --s3-bucket or --stereo-dir")
        return 1
    logger.info("measuring pitch for %d episode(s)", len(tables))

    # Append as we go and skip what is already there. This is a network-bound
    # pass over 400 episodes; writing only at the end means an interruption
    # costs the whole run, which is exactly what happened once.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.out.open(encoding="utf-8"))) if args.out.exists() else []
    done = {r["wav_name"] for r in rows}
    todo = [t for t in tables if t.wav_name not in done]
    if done:
        logger.info("resuming: %d already measured, %d to go", len(done), len(todo))

    fresh = not args.out.exists()
    with args.out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if fresh:
            writer.writeheader()
        for n, table in enumerate(todo, 1):
            try:
                row = measure(table, store, args.stereo_dir, args.want_sec, args.window,
                              args.min_turn, args.workers)
            except Exception as exc:  # a failed episode must not sink the pass
                logger.warning("%s: %s", table.wav_name, exc)
                row = {"wav_name": table.wav_name, "podcast": table.podcast,
                       "episode_id": table.episode_id, "semitones": "", "note": str(exc)[:80]}
            rows.append(row)
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                logger.info("  %d/%d", n, len(todo))

    measured = [float(r["semitones"]) for r in rows if r.get("semitones") not in ("", None)]
    logger.info("wrote %s: %d/%d measured", args.out, len(measured), len(rows))
    if measured:
        measured.sort()
        q = lambda k: measured[int(k * (len(measured) - 1))]  # noqa: E731
        logger.info("  semitones  p10=%.1f  median=%.1f  p90=%.1f", q(0.1), q(0.5), q(0.9))
        edges = [0, 1, 2, 3, 4, 6, 8, 12, 99]
        for lo, hi in zip(edges, edges[1:]):
            n = sum(1 for s in measured if lo <= s < hi)
            logger.info("  %2d-%2d st %s %d", lo, hi, "#" * (60 * n // len(measured)), n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
