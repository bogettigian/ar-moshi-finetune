from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.config
from pathlib import Path

import sphn

logger = logging.getLogger(__name__)


def split_bucket(path: Path, val_fraction: float) -> str:
    digest = hashlib.md5(path.name.encode()).digest()
    bucket = digest[0] / 255.0
    return "val" if bucket < val_fraction else "train"


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stereo-dir", type=Path, default=Path("./data/stereo"))
    parser.add_argument("--out-dir", type=Path, default=Path("./data/jsonl"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="If set, store relative paths in the jsonl (relative to this dir).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(args.stereo_dir.glob("*.wav"))
    logger.info("found %d wavs under %s", len(wavs), args.stereo_dir)
    if not wavs:
        logger.info("no wavs found, nothing to do")
        return

    str_paths = [str(p) for p in wavs]
    durations = sphn.durations(str_paths)

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"

    train_count = 0
    val_count = 0
    with train_path.open("w") as ftrain, val_path.open("w") as fval:
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
