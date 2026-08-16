from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import sphn

from common import atomic_write, setup_logging

logger = logging.getLogger(__name__)


def split_bucket(path: Path, val_fraction: float) -> str:
    digest = hashlib.md5(path.name.encode()).digest()
    bucket = digest[0] / 255.0
    return "val" if bucket < val_fraction else "train"


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-dir", type=Path, default=Path("./data/stereo"))
    parser.add_argument("--out-dir", type=Path, default=Path("./data/jsonl"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--path-prefix",
        type=Path,
        default=Path("data/stereo"),
        help="Where the wavs will live on the training machine. Only used when "
        "building from the index, since the files may not be local here.",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="If set, store relative paths in the jsonl (relative to this dir).",
    )
    parser.add_argument(
        "--require-alignments",
        action="store_true",
        help="Fail unless every wav has its <wav>.json alongside. The trainer "
        "opens that file for every sample it draws, so without this a corpus "
        "that was never annotated only fails once training has started.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The pipeline deletes each wav once it is in S3, so the index is the only
    # local record of what the corpus contains. Fall back to scanning the
    # directory when running without S3 (smoke tests, local experiments).
    # One index per worker (index-0.csv, index-1.csv, ...), plus any index.csv
    # left by an earlier single-process run. Reading all of them is what keeps
    # the episodes from those runs in the corpus.
    index_paths = sorted(args.stereo_dir.glob("index*.csv"))
    if index_paths:
        rows = []
        for path in index_paths:
            with path.open(newline="", encoding="utf-8") as fh:
                rows.extend(csv.DictReader(fh))
        rows = sorted(rows, key=lambda r: r["wav_name"])
        logger.info(
            "found %d entries across %d index file(s)", len(rows), len(index_paths)
        )
        wavs = [args.path_prefix / r["wav_name"] for r in rows]
        durations = [float(r["duration_sec"]) for r in rows]
    else:
        wavs = sorted(args.stereo_dir.glob("*.wav"))
        logger.info("found %d wavs under %s", len(wavs), args.stereo_dir)
        durations = sphn.durations([str(p) for p in wavs]) if wavs else []

    if not wavs:
        logger.info("nothing to do")
        return

    if args.require_alignments:
        missing = [w for w in wavs if not w.with_suffix(".json").exists()]
        if missing:
            sample = ", ".join(w.name for w in missing[:5])
            raise SystemExit(
                f"{len(missing)} of {len(wavs)} wavs have no alignment json "
                f"(e.g. {sample}). Run the annotate stage before training."
            )
        logger.info("all %d wavs have their alignment json", len(wavs))

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"

    train_count = 0
    val_count = 0
    with atomic_write(train_path) as ftrain, atomic_write(val_path) as fval:
        for wav, duration in zip(wavs, durations):
            if duration is None:
                logger.warning("skip %s (could not read duration)", wav.name)
                continue
            bucket = split_bucket(wav, args.val_fraction)
            if args.relative_to is not None:
                path_field = str(wav.relative_to(args.relative_to))
            else:
                path_field = str(wav)
            line = json.dumps({"path": path_field, "duration": duration})
            if bucket == "val":
                fval.write(line + "\n")
                val_count += 1
            else:
                ftrain.write(line + "\n")
                train_count += 1

    logger.info(f"wrote {train_count} train + {val_count} val entries to {args.out_dir}")


if __name__ == "__main__":
    main()
