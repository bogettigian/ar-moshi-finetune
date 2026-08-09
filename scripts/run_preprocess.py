from __future__ import annotations

import argparse
import csv
import logging
import logging.config
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FrameType

from build_stereo import build_one
from common import (
    atomic_write,
    parse_rttm,
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


INDEX_NAME = "index.csv"
INDEX_KEY = f"stereo/{INDEX_NAME}"
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
    """Index the wav, upload it, and only then drop the local copy.

    Deleting after a successful upload is what makes a failed upload safe: the
    wav survives and the next run picks it up again.
    """
    wav_name = stereo_path.name

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
        write_index(args.stereo_dir / INDEX_NAME, index)

    store.upload(args.stereo_dir / INDEX_NAME, INDEX_KEY)
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

    audio_path = download(entry, args.raw_dir, timeout=args.timeout)
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
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Durable store. Omitted means local-only, used by the smoke test.",
    )
    parser.add_argument("--s3-prefix", default="corpus")
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Keep the downloaded audio. Off by default: it is re-downloadable "
        "from the feed and the stereo wav is what training consumes.",
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
    index_path = args.stereo_dir / INDEX_NAME
    if not index_path.exists() and INDEX_KEY in done_keys:
        logger.info("restoring %s from s3", INDEX_NAME)
        store.download(INDEX_KEY, index_path)
    index = load_index(index_path)
    indexed = {row["wav_name"] for row in index}

    seen = read_manifest_keys(args.raw_dir)
    feeds = read_feeds_file(args.feeds_file)
    logger.info(
        "%d feeds, %d episodes already in the manifest, s3=%s",
        len(feeds),
        len(seen),
        "on" if store.enabled else "off",
    )

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning("no HUGGINGFACE_TOKEN / HF_TOKEN in env — pyannote may fail")
    pipeline = load_pipeline(args.model, resolve_device(args.device), hf_token)
    counters = Counters()

    for feed_url in feeds:
        if _stop:
            break
        try:
            entries = parse_feed(feed_url)
        except Exception:
            logger.exception("failed to parse feed %s", feed_url)
            continue
        for entry in entries[: args.max_episodes_per_feed]:
            if _stop:
                break
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

    # Re-upload both records at the end: if an index upload failed mid-run the
    # wav is already in S3, and this closes the gap without extra bookkeeping.
    manifest = args.raw_dir / "manifest.csv"
    if manifest.exists():
        store.upload(manifest, "manifest.csv")
    if index_path.exists():
        store.upload(index_path, INDEX_KEY)

    logger.info(
        f"built {counters.done}, rejected {counters.rejected}, "
        f"skipped {counters.skipped}, failed {counters.failed}"
    )
    if _stop:
        logger.warning("stopped early; re-run the same command to resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
