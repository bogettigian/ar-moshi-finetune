from __future__ import annotations

import logging
import os
import resource
import shutil
import struct
import subprocess
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

logger = logging.getLogger(__name__)


def silence_audio_backend_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*MPEG_LAYER_III subtype is unknown to TorchAudio.*",
        category=UserWarning,
    )

AUDIO_EXTS = (".mp3", ".m4a")

DECODED_EXT = ".wav"

# Appended after the full filename, so `ep.wav.tmp-123` never matches a
# `*.wav` glob and half-written files stay invisible to the next stage.
TMP_SUFFIX = ".tmp-"


def iter_decoded_files(root: Path) -> list[Path]:
    return sorted(root.rglob(f"*{DECODED_EXT}"))


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}{TMP_SUFFIX}{os.getpid()}")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_write(path: Path, mode: str = "w", **kwargs) -> Iterator[IO]:
    with atomic_path(path) as tmp:
        with tmp.open(mode, **kwargs) as fh:
            yield fh


class DecodeFailed(RuntimeError):
    pass


def decode_to_wav(src: Path, dest: Path, sample_rate: int) -> Path:
    if shutil.which("ffmpeg") is None:
        raise DecodeFailed("ffmpeg is not on PATH; it decodes every episode")

    with atomic_path(dest) as tmp:
        result = subprocess.run(
            [
                "ffmpeg",
                # cloud-init gives the process no usable stdin, and ffmpeg
                # otherwise waits on it before overwriting anything.
                "-nostdin",
                "-v", "error",
                "-i", str(src),
                "-ac", "1",
                "-ar", str(sample_rate),
                "-c:a", "pcm_s16le",
                # Named explicitly: atomic_path hands us a `.tmp-<pid>` name, so
                # there is no extension left for ffmpeg to pick a muxer from.
                "-f", "wav",
                "-y", str(tmp),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DecodeFailed(
                f"ffmpeg failed on {src.name} ({result.returncode}): "
                f"{result.stderr.strip()}"
            )
    return dest


def _current_rss_kb() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass

    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
        )
        return int(out.stdout.strip())
    except (OSError, ValueError):
        return None


def rss_gb() -> tuple[float, float]:
    scale = 1 << 20 if sys.platform.startswith("linux") else 1 << 30
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale

    current_kb = _current_rss_kb()
    if current_kb is None:
        logger.warning("cannot read resident memory on this platform")
        return float("nan"), peak
    return current_kb / (1 << 20), peak


class MemTrace:
    def __init__(self) -> None:
        self.start = rss_gb()[0]
        self.marks: list[tuple[str, float]] = []

    def mark(self, stage: str) -> None:
        self.marks.append((stage, rss_gb()[0]))

    def summary(self) -> str:
        parts = [f"start {self.start:.1f}"]
        previous = self.start
        for stage, current in self.marks:
            parts.append(f"{stage} {current - previous:+.1f} -> {current:.1f}")
            previous = current
        return f"{' | '.join(parts)} | peak {rss_gb()[1]:.1f}"


def sweep_temp_files(root: Path) -> int:
    removed = 0
    for tmp in root.rglob(f"*{TMP_SUFFIX}*"):
        tmp.unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.info("removed %d orphaned temp file(s) under %s", removed, root)
    return removed


@dataclass(frozen=True)
class Segment:
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def parse_rttm(rttm_path: Path) -> list[Segment]:
    segments: list[Segment] = []
    for line in rttm_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        segments.append(
            Segment(
                start=float(parts[3]),
                duration=float(parts[4]),
                speaker=parts[7],
            )
        )
    return segments


def wav_duration_sec(path: Path) -> float:
    header = path.read_bytes()[:44]
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError(f"not a RIFF/WAVE file: {path}")
    byte_rate = struct.unpack("<I", header[28:32])[0]
    if byte_rate == 0:
        raise ValueError(f"zero byte rate in header: {path}")
    return (path.stat().st_size - 44) / byte_rate


def speaker_changes_per_min(segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    ordered = sorted(segments, key=lambda s: s.start)
    changes = sum(
        1 for a, b in zip(ordered, ordered[1:]) if a.speaker != b.speaker
    )
    span_min = (max(s.end for s in ordered) - min(s.start for s in ordered)) / 60
    return changes / span_min if span_min > 0 else 0.0


def count_dominant_speakers(segments: list[Segment], min_share: float = 0.1) -> int:
    speaker_time: dict[str, float] = defaultdict(float)
    total_time = 0.0
    for seg in segments:
        speaker_time[seg.speaker] += seg.duration
        total_time += seg.duration
    if total_time == 0:
        return 0
    return sum(1 for t in speaker_time.values() if t / total_time >= min_share)
