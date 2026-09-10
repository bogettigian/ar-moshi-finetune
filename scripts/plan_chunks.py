from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import turns as turntable
from common import atomic_write, intersect_duration, setup_logging, union_duration

logger = logging.getLogger(__name__)

EPISODE_COLUMNS = [
    "wav_name", "podcast", "duration_sec", "top2_share_seg", "coverage",
    "turn_count", "changes_per_min", "overlap_share", "speaker_balance",
    "semitones", "kept", "reject_reason", "chunks", "chunk_sec", "speech_sec", "window_sec",
]


@dataclass
class Chunk:
    start: float
    end: float
    n_turns: int
    changes: int
    seconds_ch0: float
    seconds_ch1: float
    speech_sec: float
    overlap_sec: float
    max_internal_gap: float

    @property
    def duration(self) -> float:
        """Wall clock. NOT the same as speech -- see `speech_sec`."""
        return self.end - self.start


def split_runs(table: turntable.TurnTable, max_gap: float) -> list[list[turntable.Turn]]:
    runs: list[list[turntable.Turn]] = []
    current: list[turntable.Turn] = []
    reach = -math.inf
    for turn in table.turns:
        if current and turn.start - reach > max_gap:
            runs.append(current)
            current = []
            reach = -math.inf
        current.append(turn)
        reach = max(reach, turn.end)
    if current:
        runs.append(current)
    return runs


def blocks(run: list[turntable.Turn]) -> list[tuple[list[turntable.Turn], float, float]]:
    out: list[tuple[list[turntable.Turn], float, float]] = []
    current = [run[0]]
    start, reach = run[0].start, run[0].end
    for turn in run[1:]:
        if turn.start > reach:
            out.append((current, start, reach))
            current, start, reach = [turn], turn.start, turn.end
        else:
            current.append(turn)
            reach = max(reach, turn.end)
    out.append((current, start, reach))
    return out


def cut_run(
    run: list[turntable.Turn], min_chunk: float, max_chunk: float,
    min_changes: int, min_share: float,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    atoms = blocks(run)
    i = 0
    while i < len(atoms):
        j = i
        start, end = atoms[i][1], atoms[i][2]
        while j + 1 < len(atoms) and atoms[j + 1][2] - start <= max_chunk:
            j += 1
            end = atoms[j][2]
        if end - start > max_chunk:
            # A single block longer than the window: an unbroken stretch of
            # speech with no gap to cut at. Splitting it would clip a turn, and
            # keeping it would hand the dataloader a file it re-cuts blindly --
            # the exact thing this planner exists to prevent. Skip it.
            i += 1
            continue
        if end - start < min_chunk:
            i += 1
            continue
        span = [turn for atom in atoms[i:j + 1] for turn in atom[0]]
        ivs0 = [(t.start, t.end) for t in span if t.channel == 0]
        ivs1 = [(t.start, t.end) for t in span if t.channel == 1]
        ch0, ch1 = union_duration(ivs0), union_duration(ivs1)
        # Cross-channel union, so overlapped speech counts once. Summing the two
        # channels instead would double-count it, which is how a chunk ends up
        # looking fuller than it is.
        speech = union_duration(ivs0 + ivs1)
        overlap = intersect_duration(ivs0, ivs1)
        ordered = sorted(span, key=lambda t: (t.start, t.channel))
        changes = sum(1 for a, b in zip(ordered, ordered[1:]) if a.channel != b.channel)
        gap = max((b[1] - a[2] for a, b in zip(atoms[i:j + 1], atoms[i + 1:j + 1])), default=0.0)
        minority = min(ch0, ch1) / max(ch0 + ch1, 1e-9)
        if ch0 > 0 and ch1 > 0 and changes >= min_changes and minority >= min_share:
            chunks.append(
                Chunk(start, end, len(span), changes, ch0, ch1, speech, overlap, gap)
            )
            i = j + 1
        else:
            i += 1
    return chunks


def plan_episode(table, max_gap, min_chunk, max_chunk, min_changes, min_share) -> list[Chunk]:
    out: list[Chunk] = []
    for run in split_runs(table, max_gap):
        out += cut_run(run, min_chunk, max_chunk, min_changes, min_share)
    return out


def read_pitch(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    return {
        r["wav_name"]: float(r["semitones"])
        for r in csv.DictReader(path.open(encoding="utf-8"))
        if r.get("semitones") not in ("", None)
    }


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns-dir", type=Path, default=Path("./data/turns"))
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="Limit to these wav_names. For the 'run on one or two "
                             "representative episodes first' review.")
    parser.add_argument("--pitch-csv", type=Path, default=None,
                        help="Output of voice_pitch.py. Episodes missing from it are "
                             "KEPT with a warning -- a failed measurement must never "
                             "silently delete data.")
    parser.add_argument("--min-top2-share", type=float, default=0.90)
    parser.add_argument("--min-semitones", type=float, default=0.0,
                        help="Report-only at 0. Pick the real threshold off the "
                             "histogram, not off a round number.")
    parser.add_argument("--max-internal-gap", type=float, default=2.0)
    parser.add_argument("--min-chunk", type=float, default=30.0)
    parser.add_argument("--max-chunk", type=float, default=100.0,
                        help="Must equal the config's duration_sec (default: %(default)s).")
    parser.add_argument("--min-changes", type=int, default=2)
    parser.add_argument("--min-speaker-share", type=float, default=0.15)
    parser.add_argument("--sweep", action="store_true",
                        help="Grid over max-internal-gap x max-chunk and log the hours "
                             "table instead of writing a plan.")
    parser.add_argument("--out", type=Path, default=Path("./data/chunks/plan.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("./reports/chunk_plan.csv"))
    args = parser.parse_args()

    paths = sorted(args.turns_dir.glob("*.turns.json"))
    tables = [turntable.load(p) for p in paths]
    if args.episodes:
        wanted = set(args.episodes)
        tables = [t for t in tables if t.wav_name in wanted or t.episode_id in wanted]
    if not tables:
        logger.error("no turn tables under %s", args.turns_dir)
        return 1
    pitch = read_pitch(args.pitch_csv)
    logger.info("%d episode(s), %d with a pitch measurement", len(tables), len(pitch))

    def keep(table) -> str | None:
        if float(table.source.get("top2_share_seg", 1.0)) < args.min_top2_share:
            return "top2_share"
        semis = pitch.get(table.wav_name)
        if semis is None and pitch:
            logger.warning("no pitch row for %s, keeping it", table.wav_name)
        elif semis is not None and semis < args.min_semitones:
            return "semitones"
        return None

    if args.sweep:
        logger.info(
            "surviving hours by max pause (rows) and chunk length (cols): "
            "wall-clock/speech, then speech as a share of the training window"
        )
        # min-chunk is held at its flag value, so a column equal to it is
        # degenerate: a chunk would have to land exactly on a turn boundary.
        lengths = [45.0, 60.0, 100.0, 120.0]
        logger.info("%s", f"{'pause':>8}" + "".join(f"{f'{L:.0f}s':>18}" for L in lengths))
        for gap in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            cells = []
            for length in lengths:
                wall, speech, count = 0.0, 0.0, 0
                for table in tables:
                    if keep(table):
                        continue
                    for c in plan_episode(table, gap, args.min_chunk, length,
                                          args.min_changes, args.min_speaker_share):
                        wall += c.duration
                        speech += c.speech_sec
                        count += 1
                window = count * length
                cells.append(
                    f"{wall/3600:6.0f}/{speech/3600:5.0f}h {100*speech/max(window,1):3.0f}%"
                )
            logger.info("%s", f"{gap:7.1f}s" + "".join(f"{c:>20}" for c in cells))
        return 0

    rows, planned = [], []
    for table in tables:
        stats = table.stats()
        reason = keep(table)
        chunks = [] if reason else plan_episode(
            table, args.max_internal_gap, args.min_chunk, args.max_chunk,
            args.min_changes, args.min_speaker_share)
        balance = min(stats["seconds_ch0"], stats["seconds_ch1"]) / max(
            stats["seconds_ch0"] + stats["seconds_ch1"], 1e-9)
        rows.append({
            "wav_name": table.wav_name, "podcast": table.podcast,
            "duration_sec": table.duration_sec,
            "top2_share_seg": table.source.get("top2_share_seg"),
            "coverage": stats["coverage"], "turn_count": stats["turn_count"],
            "changes_per_min": stats["changes_per_min"],
            "overlap_share": stats["overlap_share"], "speaker_balance": round(balance, 4),
            "semitones": pitch.get(table.wav_name, ""),
            "kept": reason is None, "reject_reason": reason or "",
            "chunks": len(chunks), "chunk_sec": round(sum(c.duration for c in chunks), 1),
            "speech_sec": round(sum(c.speech_sec for c in chunks), 1),
            "window_sec": round(len(chunks) * args.max_chunk, 1),
        })
        for n, c in enumerate(chunks):
            planned.append({
                "wav_name": table.wav_name,
                "chunk_id": f"{table.episode_id}__{n:04d}",
                "start": round(c.start, 3), "end": round(c.end, 3),
                "duration": round(c.duration, 3), "n_turns": c.n_turns,
                "changes": c.changes,
                "seconds_ch0": round(c.seconds_ch0, 2),
                "seconds_ch1": round(c.seconds_ch1, 2),
                "speech_sec": round(c.speech_sec, 2),
                "overlap_sec": round(c.overlap_sec, 2),
                "max_internal_gap": round(c.max_internal_gap, 3),
            })

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(args.report, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(args.out) as fh:
        for entry in planned:
            fh.write(json.dumps(entry) + "\n")

    kept = [r for r in rows if r["kept"]]
    wall = sum(r["chunk_sec"] for r in rows) / 3600
    speech = sum(r["speech_sec"] for r in rows) / 3600
    window = len(planned) * args.max_chunk / 3600
    mean_len = sum(c["duration"] for c in planned) / len(planned) if planned else 0.0
    logger.info("wrote %s and %s", args.report, args.out)
    logger.info(
        "%d/%d episode(s) kept, %d chunk(s), mean %.0f s "
        "(max pause %.1fs, chunks %.0f-%.0fs)",
        len(kept), len(rows), len(planned), mean_len,
        args.max_internal_gap, args.min_chunk, args.max_chunk,
    )
    # Three different hours, and they get confused for each other. The window is
    # what the GPU pays for; the wall clock is what compares against the corpus
    # targets, which count recording duration; the speech is what is actually
    # said. Report all three so the next reader does not have to ask.
    logger.info("  training window   %6.1f h   (%.0f s x %d, what the GPU costs)",
                window, args.max_chunk, len(planned))
    logger.info("  chunk wall-clock  %6.1f h   %3.0f%% of window, padding is masked out",
                wall, 100 * wall / max(window, 1e-9))
    logger.info("  speech            %6.1f h   %3.0f%% of window, %3.0f%% of chunk",
                speech, 100 * speech / max(window, 1e-9), 100 * speech / max(wall, 1e-9))
    for reason in sorted({r["reject_reason"] for r in rows if r["reject_reason"]}):
        n = sum(1 for r in rows if r["reject_reason"] == reason)
        logger.info("  rejected by %s: %d episode(s)", reason, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
