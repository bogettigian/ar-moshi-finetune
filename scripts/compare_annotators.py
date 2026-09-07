from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import re
import statistics
from pathlib import Path

import numpy as np
import sphn

from common import setup_logging

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000

FILLERS = {
    "eh", "ehh", "ehhh", "em", "mm", "mmm", "mmhm", "ajá", "aja", "ah", "ahh",
    "uh", "uhh", "hm", "hmm", "mhm", "ey", "esteee", "ehm",
}
WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize(word: str) -> str:
    return WORD_RE.sub("", word.lower())


def load_alignments(path: Path) -> tuple[list[tuple[str, float, float]], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    words = [(w, float(a), float(b)) for w, (a, b), _ in data["alignments"]]
    return words, data.get("alignment_stats", {})


def silence_mask(wav: Path, channel: int) -> np.ndarray:
    audio, _ = sphn.read(str(wav), sample_rate=SAMPLE_RATE)
    return audio[channel] == 0.0


def words_on_silence(words, mask: np.ndarray) -> int:
    misplaced = 0
    for _, start, end in words:
        middle = int((start + end) / 2 * SAMPLE_RATE)
        if 0 <= middle < len(mask) and mask[middle]:
            misplaced += 1
    return misplaced


def text_metrics(words) -> dict:
    tokens = [normalize(w) for w, _, _ in words]
    tokens = [t for t in tokens if t]
    speech = sum(end - start for _, start, end in words)
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return {
        "words": len(tokens),
        "unique_words": len(set(tokens)),
        "fillers": sum(1 for t in tokens if t in FILLERS),
        "immediate_repeats": repeats,
        "spoken_seconds": round(speech, 1),
    }


def timing_disagreement(left, right) -> float | None:
    a = [normalize(w) for w, _, _ in left]
    b = [normalize(w) for w, _, _ in right]
    gaps = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag != "equal":
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            gaps.append(abs(left[i][1] - right[j][1]))
    if not gaps:
        return None
    return round(1000 * statistics.median(gaps), 1)


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="NAME=directory with the alignment jsons.")
    parser.add_argument("--stereo-dir", type=Path, required=True, help="Where the wavs are, to check timestamps against digital silence.")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("./reports/annotator_comparison.csv"))
    args = parser.parse_args()

    runs: dict[str, Path] = {}
    for entry in args.runs:
        name, _, directory = entry.partition("=")
        if not directory:
            parser.error(f"expected NAME=directory, got {entry!r}")
        runs[name] = Path(directory)

    episodes = sorted({p.stem for d in runs.values() for p in d.glob("*.json")})
    if not episodes:
        logger.error("no alignment jsons under %s", ", ".join(str(d) for d in runs.values()))
        return 1
    logger.info("%d episode(s) x %d configuration(s)", len(episodes), len(runs))

    masks = {e: silence_mask(args.stereo_dir / f"{e}.wav", args.channel) for e in episodes}
    loaded: dict[tuple[str, str], list] = {}
    rows: list[dict] = []

    for name, directory in runs.items():
        for episode in episodes:
            path = directory / f"{episode}.json"
            if not path.exists():
                logger.warning("%s has no output for %s", name, episode)
                continue
            words, stats = load_alignments(path)
            loaded[(name, episode)] = words
            metrics = text_metrics(words)
            misplaced = words_on_silence(words, masks[episode])
            # Per minute of speech in the episode, not per minute of transcribed
            # words: the latter is a different denominator for every config --
            # a system whose words carry longer durations would look slower --
            # and the whole point is to compare them against the same audio.
            speech = stats.get("speech_seconds") or metrics["spoken_seconds"]
            minutes = max(float(speech), 1e-9) / 60
            rows.append(
                {
                    "config": name,
                    "episode": episode,
                    **metrics,
                    "speech_seconds": round(float(speech), 1),
                    "words_per_min": round(metrics["words"] / minutes, 1),
                    "fillers_per_min": round(metrics["fillers"] / minutes, 2),
                    "words_on_silence": misplaced,
                    "pct_on_silence": round(100 * misplaced / max(metrics["words"], 1), 2),
                    "score_median": stats.get("score_median", ""),
                    "words_without_timing": stats.get("words_without_timing", ""),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Written to %s", args.out)

    header = f"{'config':<22}{'words':>7}{'w/min':>8}{'fillers':>9}{'f/min':>7}{'repeats':>9}{'on_silence':>12}{'score':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for name in runs:
        mine = [r for r in rows if r["config"] == name]
        if not mine:
            continue
        total = sum(r["words"] for r in mine)
        fillers = sum(r["fillers"] for r in mine)
        silent = sum(r["words_on_silence"] for r in mine)
        minutes = sum(r["speech_seconds"] for r in mine) / 60
        scores = [float(r["score_median"]) for r in mine if r["score_median"] not in ("", None)]
        logger.info(
            "%s",
            f"{name:<22}{total:>7}{total/max(minutes,1e-9):>8.0f}{fillers:>9}"
            f"{fillers/max(minutes,1e-9):>7.2f}{sum(r['immediate_repeats'] for r in mine):>9}"
            f"{silent:>7} ({100*silent/max(total,1):>3.0f}%)"
            f"{statistics.fmean(scores) if scores else float('nan'):>8.2f}",
        )

    names = list(runs)
    if len(names) > 1:
        logger.info("median timing disagreement, in ms, over the words both transcribed:")
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                gaps = [
                    timing_disagreement(loaded[(left, e)], loaded[(right, e)])
                    for e in episodes
                    if (left, e) in loaded and (right, e) in loaded
                ]
                gaps = [g for g in gaps if g is not None]
                if gaps:
                    logger.info("  %10s vs %-10s %7.0f ms", left, right, statistics.fmean(gaps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
