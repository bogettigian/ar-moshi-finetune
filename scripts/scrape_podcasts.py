from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import logging.config
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import requests

from common import AUDIO_EXTS, atomic_path, sweep_temp_files

logger = logging.getLogger(__name__)

MANIFEST_COLUMNS = (
    "podcast",
    "episode_id",
    "title",
    "published",
    "duration_sec_estimated",
    "url",
    "local_path",
)


@dataclass(frozen=True)
class FeedEntry:
    podcast: str
    episode_id: str
    title: str
    published: str
    duration_sec_estimated: float | None
    url: str
    ext: str


# Many feeds mix MP3 and M4A within the same channel. Storing everything as
# .mp3 produces files that soundfile cannot open and that look corrupt without
# being so, so the extension comes from the enclosure.
EXT_BY_MIME = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".m4a",
}


def pick_extension(mime: str | None, url: str) -> str:
    ext = EXT_BY_MIME.get((mime or "").split(";")[0].strip().lower())
    if ext:
        return ext
    url_ext = Path(urlparse(url).path).suffix.lower()
    return url_ext if url_ext in AUDIO_EXTS else ".mp3"


def parse_feed(url: str) -> list[FeedEntry]:
    logger.info("parsing feed %s", url)
    parsed = feedparser.parse(url)
    podcast_title = parsed.feed.get("title") or urlparse(url).netloc
    podcast_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", podcast_title).strip("-").lower()[:64] or "unnamed"

    entries: list[FeedEntry] = []
    for entry in parsed.entries:
        enclosures = [e for e in entry.get("enclosures", []) if e.get("href")]
        audio = next(
            (e for e in enclosures if (e.get("type") or "").startswith("audio")),
            enclosures[0] if enclosures else None,
        )
        if audio is None:
            continue
        enclosure_url = audio["href"]
        title = entry.get("title", "untitled")
        published = entry.get("published", "")
        episode_id = entry.get("id") or enclosure_url
        episode_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", title).strip("-").lower()[:64] or hashlib.md5(episode_id.encode()).hexdigest()[:16]
        duration_sec = _parse_duration(entry.get("itunes_duration"))
        entries.append(
            FeedEntry(
                podcast=podcast_slug,
                episode_id=episode_slug,
                title=title,
                published=published,
                duration_sec_estimated=duration_sec,
                url=enclosure_url,
                ext=pick_extension(audio.get("type"), enclosure_url),
            )
        )
    logger.info("feed %s: %d episodes", podcast_slug, len(entries))
    return entries


def _parse_duration(raw: str | None) -> float | None:
    if not raw:
        return None
    parts = raw.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
    elif len(nums) == 2:
        h, m, s = 0, nums[0], nums[1]
    elif len(nums) == 1:
        h, m, s = 0, 0, nums[0]
    else:
        return None
    return float(h * 3600 + m * 60 + s)


class TruncatedDownload(RuntimeError):
    pass


def _fetch(url: str, dest: Path, timeout: float) -> None:
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        expected = resp.headers.get("Content-Length")
        written = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    written += fh.write(chunk)
    if expected is None:
        logger.warning("no Content-Length for %s, cannot verify size", url)
        return
    if written != int(expected):
        raise TruncatedDownload(f"got {written} bytes, expected {expected}")


def download(
    entry: FeedEntry, out_dir: Path, timeout: float = 60.0, attempts: int = 3
) -> Path:
    podcast_dir = out_dir / entry.podcast
    podcast_dir.mkdir(parents=True, exist_ok=True)
    local_path = podcast_dir / f"{entry.episode_id}{entry.ext}"
    # Check every known extension: an episode downloaded before the scraper
    # told formats apart may sit on disk as .mp3 even though the feed declares
    # it M4A. Do not re-download it.
    for ext in AUDIO_EXTS:
        existing = local_path.with_suffix(ext)
        if existing.exists():
            logger.debug("skip (exists): %s", existing)
            return existing
    logger.info("downloading %s in %s", entry.url, local_path)
    for attempt in range(1, attempts + 1):
        try:
            with atomic_path(local_path) as tmp:
                _fetch(entry.url, tmp, timeout)
            return local_path
        except (requests.RequestException, TruncatedDownload) as exc:
            if attempt == attempts:
                raise
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "attempt %d/%d failed for %s (%s), retrying in %ds",
                attempt,
                attempts,
                entry.url,
                exc,
                backoff,
            )
            time.sleep(backoff)
    return local_path


def read_manifest_keys(out_dir: Path) -> set[tuple[str, str]]:
    manifest_path = out_dir / "manifest.csv"
    if not manifest_path.exists():
        return set()
    with manifest_path.open(newline="") as fh:
        return {(row["podcast"], row["episode_id"]) for row in csv.DictReader(fh)}


def append_manifest(
    out_dir: Path, entry: FeedEntry, local_path: Path, seen: set[tuple[str, str]]
) -> None:
    key = (entry.podcast, entry.episode_id)
    if key in seen:
        logger.debug("skip manifest (already listed): %s", local_path)
        return
    manifest_path = out_dir / "manifest.csv"
    is_new = not manifest_path.exists()
    with manifest_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(
            (
                entry.podcast,
                entry.episode_id,
                entry.title,
                entry.published,
                entry.duration_sec_estimated or "",
                entry.url,
                str(local_path),
            )
        )
    seen.add(key)


def read_feeds_file(path: Path) -> list[str]:
    feeds: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        feeds.append(line)
    return feeds


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feeds-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("./data/raw_mp3"))
    parser.add_argument(
        "--max-episodes-per-feed",
        type=int,
        default=50,
        help="Newest N episodes to take from each feed (default: %(default)s).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_temp_files(args.out_dir)
    feeds = read_feeds_file(args.feeds_file)
    logger.info("loaded %d feeds from %s", len(feeds), args.feeds_file)

    seen = read_manifest_keys(args.out_dir)
    logger.info("manifest already lists %d episodes", len(seen))

    total = 0
    for feed_url in feeds:
        try:
            entries = parse_feed(feed_url)
        except Exception:
            logger.exception("failed to parse feed %s", feed_url)
            continue
        for entry in entries[: args.max_episodes_per_feed]:
            try:
                local_path = download(entry, args.out_dir, timeout=args.timeout)
                append_manifest(args.out_dir, entry, local_path, seen)
                total += 1
            except Exception:
                logger.exception("failed to download %s", entry.url)

    logger.info(f"downloaded {total} episodes to {args.out_dir}")


if __name__ == "__main__":
    main()
