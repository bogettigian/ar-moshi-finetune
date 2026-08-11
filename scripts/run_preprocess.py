from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import logging.config
import os
import shutil
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from build_stereo import build_one
from common import (
    DECODED_EXT,
    atomic_write,
    decode_to_wav,
    parse_rttm,
    rss_gb,
    silence_audio_backend_warnings,
    speaker_changes_per_min,
    sweep_temp_files,
    wav_duration_sec,
)
from diarize_pyannote import diarize_one, load_pipeline, resolve_device
from s3 import S3Store
from scrape_podcasts import (
    FeedEntry,
    append_manifest,
    download,
    parse_feed,
    read_feeds_file,
    read_manifest_keys,
)

logger = logging.getLogger(__name__)

_stop = False


def _request_stop(signum: int, frame: FrameType | None) -> None:
    global _stop
    if _stop:
        logger.warning("second signal %d, exiting now", signum)
        sys.exit(1)
    _stop = True
    logger.warning("signal %d received, finishing current episode then stopping", signum)


def index_name(shard: int, num_shards: int) -> str:
    return "index.csv" if num_shards <= 1 else f"index-{shard}.csv"


def in_shard(entry: FeedEntry, shard: int, num_shards: int) -> bool:
    if num_shards <= 1:
        return True
    key = f"{entry.podcast}/{entry.episode_id}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % num_shards == shard


INDEX_COLUMNS = (
    "wav_name",
    "podcast",
    "episode_id",
    "duration_sec",
    "bytes",
    "speaker_changes_per_min",
)


def load_index(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    with atomic_write(path, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Counters:
    done: int = 0
    rejected: int = 0
    failed: int = 0
    skipped: int = 0


def publish_stereo(
    entry: FeedEntry,
    stereo_path: Path,
    rttm_path: Path,
    args: argparse.Namespace,
    store: S3Store,
    done_keys: set[str],
    index: list[dict[str, str]],
    indexed: set[str],
) -> None:
    wav_name = stereo_path.name
    name = index_name(args.shard, args.num_shards)

    # Order matters. The wav goes up first so that "indexed" always implies
    # "uploaded": if this raises, nothing is recorded and the next run finds the
    # wav still on disk and retries. Indexing first would mark the episode done
    # and strand the file here forever.
    store.upload(stereo_path, f"stereo/{wav_name}")
    done_keys.add(f"stereo/{wav_name}")

    if wav_name not in indexed:
        if rttm_path.exists():
            changes = speaker_changes_per_min(parse_rttm(rttm_path))
        else:
            logger.warning("no rttm for %s, recording 0 changes/min", wav_name)
            changes = 0.0
        index.append(
            {
                "wav_name": wav_name,
                "podcast": entry.podcast,
                "episode_id": entry.episode_id,
                # Full precision: the trainer reads this straight into the
                # manifest, so it should match what reading the wav would give.
                "duration_sec": repr(wav_duration_sec(stereo_path)),
                "bytes": str(stereo_path.stat().st_size),
                "speaker_changes_per_min": f"{changes:.2f}",
            }
        )
        indexed.add(wav_name)
        write_index(args.stereo_dir / name, index)

    store.upload(args.stereo_dir / name, f"stereo/{name}")
    if store.enabled and not args.keep_stereo:
        stereo_path.unlink(missing_ok=True)


def process_episode(
    entry: FeedEntry,
    pipeline,
    args: argparse.Namespace,
    store: S3Store,
    done_keys: set[str],
    seen: set[tuple[str, str]],
    index: list[dict[str, str]],
    indexed: set[str],
    counters: Counters,
) -> None:
    wav_name = f"{entry.podcast}__{entry.episode_id}.wav"
    stereo_key = f"stereo/{wav_name}"
    stereo_path = args.stereo_dir / wav_name
    rttm_path = args.raw_dir / entry.podcast / f"{entry.episode_id}.rttm"

    if wav_name in indexed or stereo_key in done_keys:
        logger.debug("skip (already built): %s", wav_name)
        counters.skipped += 1
        return

    if stereo_path.exists():
        # Built by an earlier run whose upload never completed. Finish the job
        # instead of skipping, or the wav would sit here forever.
        logger.info("finishing upload left over from a previous run: %s", wav_name)
        publish_stereo(
            entry, stereo_path, rttm_path, args, store, done_keys, index, indexed
        )
        counters.done += 1
        return

    audio_path = args.raw_dir / entry.podcast / f"{entry.episode_id}{DECODED_EXT}"
    if audio_path.exists():
        # Left by a run that got further than the download but never finished.
        # Re-fetching and re-decoding it would be pure waste.
        logger.debug("skip download and decode (exists): %s", audio_path.name)
    else:
        source_path = download(entry, args.raw_dir, timeout=args.timeout)
        decode_to_wav(source_path, audio_path, args.sample_rate)
        # The original is dead weight from here on -- nothing downstream opens
        # it -- and keeping both would double the disk a long run needs.
        if not args.keep_source:
            source_path.unlink(missing_ok=True)

    rttm_path = diarize_one(pipeline, audio_path)
    append_manifest(args.raw_dir, entry, audio_path, seen)

    rttm_key = f"rttm/{entry.podcast}/{rttm_path.name}"
    if rttm_key not in done_keys:
        store.upload(rttm_path, rttm_key)
        done_keys.add(rttm_key)

    result = build_one(
        audio_path, rttm_path, stereo_path, args.sample_rate, args.min_share
    )
    if result is None:
        # Rejected by the 2-speaker filter. The rttm stays as the record of why.
        counters.rejected += 1
    else:
        publish_stereo(
            entry, stereo_path, rttm_path, args, store, done_keys, index, indexed
        )
        counters.done += 1

    if not args.keep_source:
        audio_path.unlink(missing_ok=True)


def main() -> int:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feeds-file", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("./data/raw_mp3"))
    parser.add_argument("--stereo-dir", type=Path, default=Path("./data/stereo"))
    parser.add_argument("--max-episodes-per-feed", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--min-share", type=float, default=0.1)
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Windows per forward pass. pyannote defaults to 1, which leaves a "
        "GPU idle; lower this when running on cpu (default: %(default)s).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Durable store. Omitted means local-only, used by the smoke test.",
    )
    parser.add_argument("--s3-prefix", default="corpus")
    parser.add_argument(
        "--shard", type=int, default=0, help="This worker's index, 0-based."
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total workers. Episodes are split by a stable hash, so every "
        "worker takes a disjoint slice (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the downloaded episode and its decoded wav. Off by default: "
        "both are reproducible from the feed and the stereo wav is what "
        "training consumes.",
    )
    parser.add_argument(
        "--keep-stereo",
        action="store_true",
        help="Keep the stereo wav locally after uploading it. Implied when no "
        "bucket is configured.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.stereo_dir.mkdir(parents=True, exist_ok=True)
    silence_audio_backend_warnings()
    sweep_temp_files(args.raw_dir)
    sweep_temp_files(args.stereo_dir)

    store = S3Store(args.s3_bucket, args.s3_prefix)
    done_keys = store.list_keys()

    # Rehydrate the index when resuming on a fresh instance: without it the
    # record of what was already built is lost and make_jsonl comes up empty.
    name = index_name(args.shard, args.num_shards)
    index_key = f"stereo/{name}"
    index_path = args.stereo_dir / name
    if not index_path.exists() and index_key in done_keys:
        logger.info("restoring %s from s3", name)
        store.download(index_key, index_path)
    index = load_index(index_path)
    indexed = {row["wav_name"] for row in index}

    seen = read_manifest_keys(args.raw_dir)
    feeds = read_feeds_file(args.feeds_file)
    # The shard goes in the log because every worker writes to the same stdout.
    logger.info(
        "shard %d/%d: %d feeds, %d episodes already in the manifest, s3=%s",
        args.shard,
        args.num_shards,
        len(feeds),
        len(seen),
        "on" if store.enabled else "off",
    )

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("no HUGGINGFACE_TOKEN / HF_TOKEN in env — pyannote may fail")
    pipeline = load_pipeline(
        args.model, resolve_device(args.device), hf_token, args.batch_size
    )
    counters = Counters()

    unusable_feeds: list[str] = []
    for feed_url in feeds:
        if _stop:
            break
        try:
            entries = parse_feed(feed_url)
        except Exception:
            logger.exception("failed to parse feed %s", feed_url)
            unusable_feeds.append(feed_url)
            continue
        if not entries:
            logger.warning("feed produced no episodes: %s", feed_url)
            unusable_feeds.append(feed_url)
        for entry in entries[: args.max_episodes_per_feed]:
            if _stop:
                break
            if not in_shard(entry, args.shard, args.num_shards):
                continue
            try:
                process_episode(
                    entry,
                    pipeline,
                    args,
                    store,
                    done_keys,
                    seen,
                    index,
                    indexed,
                    counters,
                )
            except Exception:
                logger.exception("failed on %s / %s", entry.podcast, entry.episode_id)
                counters.failed += 1

            # Logged for every episode, including the failures. Two attempts at
            # sizing this pipeline from estimates were wrong, and an OOM kill
            # only ever reports the number that finally broke it -- this is the
            # series that shows whether either resource grows across a run.
            # Disk is here because a failed episode leaves its decoded wav
            # behind, deliberately, so a retry does not download it again.
            current, peak = rss_gb()
            free = shutil.disk_usage(args.raw_dir).free / (1 << 30)
            logger.info(
                "rss %.1f GB (peak %.1f GB), disk free %.1f GB, after %s / %s (%.2f h)",
                current,
                peak,
                free,
                entry.podcast,
                entry.episode_id,
                (entry.duration_sec_estimated or 0.0) / 3600,
            )

    # Re-upload both records at the end: if an index upload failed mid-run the
    # wav is already in S3, and this closes the gap without extra bookkeeping.
    manifest = args.raw_dir / "manifest.csv"
    if manifest.exists():
        # Per-shard key for the same reason as the index: every worker keeps its
        # own local manifest, and a single shared key would keep one at random.
        suffix = "" if args.num_shards <= 1 else f"-{args.shard}"
        store.upload(manifest, f"manifest{suffix}.csv")
    if index_path.exists():
        store.upload(index_path, index_key)

    # Reported at the end and loudly: a source that was down when the run
    # started contributes nothing for the whole run, and that is easy to miss
    # among thousands of log lines.
    if unusable_feeds:
        logger.warning(
            "%d of %d feeds contributed nothing: %s",
            len(unusable_feeds),
            len(feeds),
            ", ".join(unusable_feeds),
        )

    logger.info(
        f"built {counters.done}, rejected {counters.rejected}, "
        f"skipped {counters.skipped}, failed {counters.failed}"
    )
    if _stop:
        logger.warning("stopped early; re-run the same command to resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
