from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from common import (
    assign_channels,
    atomic_write,
    count_dominant_speakers,
    intersect_duration,
    merge_intervals,
    parse_rttm,
    setup_logging,
    speaker_durations,
    top_two_speakers,
    union_duration,
)

logger = logging.getLogger(__name__)

SCHEMA = 1
DEFAULT_MERGE_GAP = 0.25
DEFAULT_MIN_SHARE = 0.1


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    channel: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TurnTable:
    wav_name: str
    podcast: str
    episode_id: str
    duration_sec: float | None
    speakers: tuple[str, str]
    turns: list[Turn]
    source: dict = field(default_factory=dict)

    def channel_intervals(self, channel: int) -> list[tuple[float, float]]:
        return [(t.start, t.end) for t in self.turns if t.channel == channel]

    def validate(self) -> None:
        merge_gap = float(self.source.get("merge_gap", 0.0))
        for a, b in zip(self.turns, self.turns[1:]):
            if (a.start, a.channel) > (b.start, b.channel):
                raise ValueError(f"{self.wav_name}: turns are not sorted")
        for t in self.turns:
            if t.end <= t.start:
                raise ValueError(f"{self.wav_name}: empty turn at {t.start}")
            if t.channel not in (0, 1):
                raise ValueError(f"{self.wav_name}: bad channel {t.channel}")
            if self.duration_sec is not None and t.end > self.duration_sec + 1.0:
                raise ValueError(f"{self.wav_name}: turn past end of audio")
        # Within a channel turns must be disjoint and separated by more than the
        # merge gap -- that is what makes the cross-channel overlap below the
        # only overlap that can exist, so measuring it is a single sweep.
        for channel in (0, 1):
            ivs = self.channel_intervals(channel)
            for a, b in zip(ivs, ivs[1:]):
                if b[0] - a[1] < merge_gap:
                    raise ValueError(
                        f"{self.wav_name}: ch{channel} turns {a} and {b} should have merged"
                    )

    def stats(self) -> dict:
        ch0, ch1 = self.channel_intervals(0), self.channel_intervals(1)
        speech = union_duration(ch0 + ch1)
        overlap = intersect_duration(ch0, ch1)
        changes = sum(1 for a, b in zip(self.turns, self.turns[1:]) if a.channel != b.channel)
        minutes = speech / 60
        return {
            "turn_count": len(self.turns),
            "speech_sec": round(speech, 2),
            "overlap_sec": round(overlap, 2),
            "overlap_share": round(overlap / speech, 4) if speech else 0.0,
            "changes_per_min": round(changes / minutes, 2) if minutes else 0.0,
            "seconds_ch0": round(union_duration(ch0), 2),
            "seconds_ch1": round(union_duration(ch1), 2),
            "coverage": round(speech / self.duration_sec, 4) if self.duration_sec else None,
        }

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "wav_name": self.wav_name,
            "podcast": self.podcast,
            "episode_id": self.episode_id,
            "duration_sec": self.duration_sec,
            "speakers": list(self.speakers),
            "source": self.source,
            # [start, end, channel]: the speaker label is recoverable from
            # `speakers[channel]`, and this is the same shape as the alignment
            # triples the trainer already consumes.
            "turns": [[round(t.start, 3), round(t.end, 3), t.channel] for t in self.turns],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(path) as fh:
            json.dump(self.to_json(), fh, ensure_ascii=False)
        return path


def load(path: Path) -> TurnTable:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {raw.get('schema')}, expected {SCHEMA}")
    table = TurnTable(
        wav_name=raw["wav_name"],
        podcast=raw["podcast"],
        episode_id=raw["episode_id"],
        duration_sec=raw["duration_sec"],
        speakers=tuple(raw["speakers"]),
        turns=[Turn(s, e, c) for s, e, c in raw["turns"]],
        source=raw.get("source", {}),
    )
    table.validate()
    return table


def build(
    rttm_path: Path,
    podcast: str,
    episode_id: str,
    duration_sec: float | None = None,
    merge_gap: float = DEFAULT_MERGE_GAP,
    min_share: float = DEFAULT_MIN_SHARE,
) -> TurnTable | None:
    segments = parse_rttm(rttm_path)
    if not segments:
        logger.warning("empty rttm: %s", rttm_path)
        return None
    if count_dominant_speakers(segments, min_share) != 2:
        return None
    try:
        top_two = top_two_speakers(segments)
    except ValueError:
        return None

    # The hash input is the DECODED source name, `<episode_id>.wav`, because that
    # is what build_stereo.build_one passed. It is not the stereo `wav_name`, and
    # the two disagree for 52% of this corpus -- getting it wrong inverts the
    # channels of half the episodes with no error anywhere.
    channel_by_speaker = assign_channels(f"{episode_id}.wav", top_two)

    turns: list[Turn] = []
    for speaker, channel in channel_by_speaker.items():
        spans = [(s.start, s.end) for s in segments if s.speaker == speaker]
        turns += [Turn(a, b, channel) for a, b in merge_intervals(spans, merge_gap)]
    turns.sort(key=lambda t: (t.start, t.channel))

    totals = speaker_durations(segments)
    table = TurnTable(
        wav_name=f"{podcast}__{episode_id}.wav",
        podcast=podcast,
        episode_id=episode_id,
        duration_sec=duration_sec,
        speakers=(top_two[0], top_two[1]) if channel_by_speaker[top_two[0]] == 0
        else (top_two[1], top_two[0]),
        turns=turns,
        source={
            "rttm": f"rttm/{podcast}/{rttm_path.name}",
            "merge_gap": merge_gap,
            "min_share": min_share,
            "top2_share_seg": round(sum(totals[s] for s in top_two) / sum(totals.values()), 4),
        },
    )
    table.validate()
    return table


def read_index(index_dir: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(index_dir.glob("index*.csv")):
        for row in csv.DictReader(path.open(encoding="utf-8")):
            rows[row["wav_name"]] = row
    return rows


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rttm-dir", type=Path, default=Path("./data/rttm"))
    parser.add_argument("--index-dir", type=Path, default=Path("./data/stereo"),
                        help="Where index*.csv lives; supplies episode duration.")
    parser.add_argument("--out-dir", type=Path, default=Path("./data/turns"))
    parser.add_argument("--merge-gap", type=float, default=DEFAULT_MERGE_GAP,
                        help="Same-speaker segments closer than this are one turn "
                             "(default: %(default)s).")
    parser.add_argument("--min-share", type=float, default=DEFAULT_MIN_SHARE,
                        help="Must match diarize_pyannote.py and build_stereo.py "
                             "(default: %(default)s).")
    parser.add_argument("--only-indexed", action="store_true",
                        help="Skip episodes with no stereo wav, i.e. the ones the "
                             "2-speaker filter already rejected.")
    args = parser.parse_args()

    index = read_index(args.index_dir)
    rttms = sorted(args.rttm_dir.rglob("*.rttm"))
    logger.info("%d rttm(s) under %s, %d indexed episode(s)", len(rttms), args.rttm_dir, len(index))

    built = skipped = rejected = 0
    for rttm in rttms:
        podcast, episode_id = rttm.parent.name, rttm.stem
        row = index.get(f"{podcast}__{episode_id}.wav")
        if args.only_indexed and row is None:
            skipped += 1
            continue
        duration = float(row["duration_sec"]) if row else None
        table = build(rttm, podcast, episode_id, duration, args.merge_gap, args.min_share)
        if table is None:
            rejected += 1
            continue
        table.save(args.out_dir / f"{podcast}__{episode_id}.turns.json")
        built += 1

    logger.info("built %d turn table(s), rejected %d, skipped %d", built, rejected, skipped)
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
