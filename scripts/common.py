from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTS = (".mp3", ".m4a")


def iter_audio_files(root: Path) -> list[Path]:
    paths = {p for ext in AUDIO_EXTS for p in root.rglob(f"*{ext}")}
    return sorted(paths)


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


def count_dominant_speakers(segments: list[Segment], min_share: float = 0.1) -> int:
    speaker_time: dict[str, float] = defaultdict(float)
    total_time = 0.0
    for seg in segments:
        speaker_time[seg.speaker] += seg.duration
        total_time += seg.duration
    if total_time == 0:
        return 0
    return sum(1 for t in speaker_time.values() if t / total_time >= min_share)
