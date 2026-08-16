from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import FrameType

from common import atomic_write, setup_logging
from s3 import S3Store

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ANNOTATE = REPO_ROOT / "annotate.py"

# Two separate flags on purpose. `_stop` means a signal arrived, and only that:
# it decides whether the workers' non-zero exits are failures and whether the
# run should be described as resumable. `_finished` only wakes the uploader,
# and is set on the way out of a healthy run too.
_stop = threading.Event()
_finished = threading.Event()
_processes: list[subprocess.Popen] = []


def _request_stop(signum: int, frame: FrameType | None) -> None:
    if _stop.is_set():
        logger.warning("second signal %d, exiting now", signum)
        sys.exit(1)
    _stop.set()
    _finished.set()
    logger.warning("signal %d received, stopping workers and shipping what is done", signum)
    # annotate.py has no signal handling of its own, so it dies mid-episode and
    # leaves a `.json.tmp.<pid>` behind. That name does not match the `*.json`
    # glob, so nothing half-written is ever uploaded or counted as done.
    for process in _processes:
        process.terminate()


def read_manifest(paths: list[Path]) -> list[tuple[Path, float]]:
    seen: dict[Path, float] = {}
    for manifest in paths:
        if not manifest.exists():
            logger.warning("no such manifest, skipping: %s", manifest)
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            seen.setdefault(Path(entry["path"]), float(entry.get("duration", 0.0)))
    return sorted(seen.items())


def shard_by_load(items: list[tuple[Path, float]], workers: int) -> list[list[Path]]:
    shards: list[list[Path]] = [[] for _ in range(workers)]
    load = [0.0] * workers
    for path, duration in sorted(items, key=lambda kv: kv[1], reverse=True):
        target = load.index(min(load))
        shards[target].append(path)
        load[target] += duration
    for i, hours in enumerate(load):
        logger.info("shard %d: %d episode(s), %.1f h", i, len(shards[i]), hours / 3600)
    return shards


def write_egs(shard: list[Path], path: Path, durations: dict[Path, float]) -> Path:
    with atomic_write(path) as fh:
        for wav in shard:
            fh.write(json.dumps({"path": str(wav), "duration": durations[wav]}) + "\n")
    return path


def upload_new(store: S3Store, stereo_dir: Path, uploaded: set[str], prefix: str) -> int:
    count = 0
    for produced in sorted(stereo_dir.glob("*.json")):
        if produced.name in uploaded:
            continue
        try:
            store.upload(produced, f"{prefix}/{produced.name}")
        except Exception:
            logger.exception("failed to upload %s, will retry next sweep", produced.name)
            continue
        uploaded.add(produced.name)
        count += 1
    return count


def ship_periodically(
    store: S3Store, stereo_dir: Path, uploaded: set[str], prefix: str, interval: float
) -> None:
    while not _finished.wait(interval):
        shipped = upload_new(store, stereo_dir, uploaded, prefix)
        if shipped:
            logger.info("uploaded %d new alignment file(s)", shipped)


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--egs",
        type=Path,
        nargs="+",
        default=[Path("./data/jsonl/train.jsonl"), Path("./data/jsonl/val.jsonl")],
        help="Manifests to annotate. Both splits by default: the trainer opens "
        "an alignment json for eval samples too.",
    )
    parser.add_argument("--stereo-dir", type=Path, default=Path("./data/stereo"))
    parser.add_argument("--work-dir", type=Path, default=Path("./data/annotate"))
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Concurrent annotate.py processes. Bound by RAM, not cores: each "
        "one reads a whole episode into memory (default: %(default)s).",
    )
    parser.add_argument("--lang", default="es")
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="Passed straight to annotate.py, which recommends medium for "
        "stereo with VAD (default: %(default)s).",
    )
    parser.add_argument("--s3-bucket", default=None, help="Omit to stay local.")
    parser.add_argument("--s3-prefix", default="corpus")
    parser.add_argument(
        "--upload-interval",
        type=float,
        default=120.0,
        help="Seconds between upload sweeps. Bounds what an interruption costs.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Annotate at most this many episodes."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Also copy the alignments produced by this run here, for comparing "
        "two Whisper sizes side by side.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete the existing json and json.err of the selected episodes "
        "first. For calibration runs; refused when a bucket is configured.",
    )
    args = parser.parse_args()

    if args.fresh and args.s3_bucket:
        parser.error("--fresh with --s3-bucket would overwrite the corpus alignments")

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    items = read_manifest(args.egs)
    if not items:
        logger.error("no episodes in %s", ", ".join(str(p) for p in args.egs))
        return 1
    durations = dict(items)

    store = S3Store(args.s3_bucket, args.s3_prefix)
    uploaded = {
        key.rsplit("/", 1)[-1]
        for key in store.list_keys("stereo/")
        if key.endswith(".json")
    }

    if args.fresh:
        for wav, _ in items:
            wav.with_suffix(".json").unlink(missing_ok=True)
            wav.with_suffix(".json.err").unlink(missing_ok=True)
        uploaded = set()

    todo = [
        (wav, duration)
        for wav, duration in items
        if f"{wav.stem}.json" not in uploaded and not wav.with_suffix(".json").exists()
    ]
    logger.info(
        "%d episode(s) in the manifest, %d already annotated, %d to do",
        len(items),
        len(items) - len(todo),
        len(todo),
    )
    if args.limit is not None:
        todo = todo[: args.limit]
        logger.info("--limit %d: annotating %d episode(s)", args.limit, len(todo))
    if not todo:
        logger.info("nothing to do")
        return 0

    workers = max(1, min(args.workers, len(todo)))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    shards = shard_by_load(todo, workers)

    shipper: threading.Thread | None = None
    if store.enabled:
        shipper = threading.Thread(
            target=ship_periodically,
            args=(store, args.stereo_dir, uploaded, "stereo", args.upload_interval),
            daemon=True,
        )
        shipper.start()

    orphans = sorted(args.stereo_dir.glob("*.json.tmp*"))
    for orphan in orphans:
        orphan.unlink(missing_ok=True)
    if orphans:
        logger.info("removed %d half-written alignment file(s)", len(orphans))

    t0 = time.time()
    for shard, wavs in enumerate(shards):
        egs = write_egs(wavs, args.work_dir / f"egs-{shard}.jsonl", durations)
        command = [
            sys.executable,
            str(ANNOTATE),
            str(egs),
            "--local",
            "--lang",
            args.lang,
            "--whisper_model",
            args.whisper_model,
        ]
        logger.info("shard %d: %s", shard, " ".join(command))
        _processes.append(subprocess.Popen(command, cwd=REPO_ROOT))

    failed = 0
    for shard, process in enumerate(_processes):
        code = process.wait()
        # After a stop signal every worker exits non-zero by definition, and
        # that is not a failure of the stage — it resumes from the store.
        if code != 0 and not _stop.is_set():
            logger.error("shard %d exited with %d", shard, code)
            failed += 1

    _finished.set()
    if shipper is not None:
        shipper.join(timeout=10)
    if store.enabled:
        logger.info("final sweep: %d file(s)", upload_new(store, args.stereo_dir, uploaded, "stereo"))

    produced = [w.with_suffix(".json") for w, _ in todo if w.with_suffix(".json").exists()]
    errors = [w for w, _ in todo if w.with_suffix(".json.err").exists()]

    if args.json_dir is not None:
        args.json_dir.mkdir(parents=True, exist_ok=True)
        for path in produced:
            (args.json_dir / path.name).write_bytes(path.read_bytes())
        logger.info("copied %d alignment file(s) to %s", len(produced), args.json_dir)

    elapsed = time.time() - t0
    audio_hours = sum(durations[w] for w, _ in todo) / 3600
    logger.info(
        "annotated %d/%d episode(s) in %.1f h of wallclock over %.1f h of audio "
        "(rtf %.3f), %d error file(s), %d failed shard(s)",
        len(produced),
        len(todo),
        elapsed / 3600,
        audio_hours,
        elapsed / (audio_hours * 3600) if audio_hours else float("nan"),
        len(errors),
        failed,
    )
    if errors:
        # A handful is normal and annotate.py keeps going; a wave of them means
        # something systematic, and the alignments would be silently incomplete.
        logger.warning(
            "episodes that failed: %s%s",
            ", ".join(w.name for w in errors[:5]),
            " ..." if len(errors) > 5 else "",
        )
    if _stop.is_set():
        logger.warning("stopped early; re-run the same command to resume")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
