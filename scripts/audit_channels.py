from __future__ import annotations

import argparse
import csv
import logging
import logging.config
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

NEAR_MONO_MIN_CORR = 0.99
SPEAKER_MAX_CORR = 0.5


def classify(
    n_channels: int, correlation: float | None, rms_diff_db: float | None
) -> str:
    if n_channels < 2:
        return "mono"
    if correlation is None:
        return "near_mono"
    if correlation >= NEAR_MONO_MIN_CORR:
        return "near_mono"
    if correlation >= SPEAKER_MAX_CORR:
        return "production_stereo"
    return "speaker_stereo"


@dataclass
class ChannelStats:
    path: str
    podcast: str
    n_channels: int
    sample_rate: int
    duration_sec: float
    lr_correlation: float | None
    lr_rms_diff_db: float | None
    classification: str


# Episodes are ~90 min, so loading a whole file at float64 costs several GB.
# Instead we stream fixed-size blocks and accumulate the second-order sums needed
# for Pearson correlation and the L-R / mix RMS ratio — bounded memory, one pass.
BLOCK_FRAMES = 1 << 20  # ~1M frames per block


def analyze(mp3_path: Path, in_dir: Path) -> ChannelStats:
    podcast = mp3_path.parent.name
    rel = str(mp3_path.relative_to(in_dir))

    info = sf.info(str(mp3_path))
    sr = info.samplerate
    n_channels = info.channels
    duration_sec = info.duration

    if n_channels < 2:
        return ChannelStats(
            path=rel,
            podcast=podcast,
            n_channels=n_channels,
            sample_rate=sr,
            duration_sec=duration_sec,
            lr_correlation=None,
            lr_rms_diff_db=None,
            classification=classify(n_channels, None, None),
        )

    n = 0
    sum_l = sum_r = sum_ll = sum_rr = sum_lr = 0.0
    for block in sf.blocks(
        str(mp3_path), blocksize=BLOCK_FRAMES, dtype="float64", always_2d=True
    ):
        left, right = block[:, 0], block[:, 1]
        n += left.shape[0]
        sum_l += float(left.sum())
        sum_r += float(right.sum())
        sum_ll += float(np.dot(left, left))
        sum_rr += float(np.dot(right, right))
        sum_lr += float(np.dot(left, right))

    # Derive everything from the accumulated sums. The per-sample 1/n cancels in
    # the ratio, so we compare summed energies directly:
    #   diff_rms^2 / mix_rms^2 = sum((L-R)^2) / sum(((L+R)/2)^2)
    diff_energy = max(sum_ll + sum_rr - 2.0 * sum_lr, 0.0)
    mix_energy = (sum_ll + sum_rr + 2.0 * sum_lr) / 4.0
    if mix_energy > 0:
        rms_diff_db = 10.0 * np.log10(max(diff_energy, 1e-24) / mix_energy)
    else:
        rms_diff_db = float("-inf")

    var_l = n * sum_ll - sum_l * sum_l
    var_r = n * sum_rr - sum_r * sum_r
    if var_l > 0 and var_r > 0:
        correlation = (n * sum_lr - sum_l * sum_r) / float(np.sqrt(var_l * var_r))
    else:
        # A channel is constant (e.g. one side silent): no linear relationship.
        correlation = 0.0

    rms_diff_db_out = round(rms_diff_db, 2) if np.isfinite(rms_diff_db) else None

    return ChannelStats(
        path=rel,
        podcast=podcast,
        n_channels=n_channels,
        sample_rate=sr,
        duration_sec=duration_sec,
        lr_correlation=round(correlation, 6),
        lr_rms_diff_db=rms_diff_db_out,
        classification=classify(n_channels, correlation, rms_diff_db_out),
    )


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("./data/raw_mp3"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("./data/raw_mp3/channel_audit.csv")
    )
    args = parser.parse_args()

    mp3_paths = sorted(args.in_dir.rglob("*.mp3"))
    logger.info("found %d mp3 files under %s", len(mp3_paths), args.in_dir)

    rows: list[ChannelStats] = []
    counts: Counter[str] = Counter()
    speaker_stereo: list[ChannelStats] = []

    for mp3_path in mp3_paths:
        try:
            stats = analyze(mp3_path, args.in_dir)
        except Exception:
            logger.exception("failed to read %s", mp3_path)
            stats = ChannelStats(
                path=str(mp3_path.relative_to(args.in_dir)),
                podcast=mp3_path.parent.name,
                n_channels=0,
                sample_rate=0,
                duration_sec=0.0,
                lr_correlation=None,
                lr_rms_diff_db=None,
                classification="error",
            )
        rows.append(stats)
        counts[stats.classification] += 1
        if stats.classification == "speaker_stereo":
            speaker_stereo.append(stats)
            logger.info(
                "speaker_stereo: %s (corr=%.4f, L-R=%.1f dB)",
                stats.path,
                stats.lr_correlation,
                stats.lr_rms_diff_db if stats.lr_rms_diff_db is not None else float("nan"),
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    logger.info("wrote catalog: %s", args.out_csv)
    logger.info(
        "summary: %s",
        ", ".join(f"{cls}={n}" for cls, n in sorted(counts.items())) or "no files",
    )
    if speaker_stereo:
        logger.warning(
            "%d episode(s) have per-speaker channel content (corr < %.2f) — "
            "candidates for classic separation; build_stereo.py:92 (arr[0]) would "
            "discard their second channel",
            len(speaker_stereo),
            SPEAKER_MAX_CORR,
        )
    else:
        logger.info(
            "no speaker-separated episodes — voices are centered; build_stereo.py's "
            "mono assumption is operationally safe (channel diff is production stereo)"
        )


if __name__ == "__main__":
    main()
