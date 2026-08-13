from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import (
    Segment,
    atomic_path,
    count_dominant_speakers,
    iter_decoded_files,
    parse_rttm,
    setup_logging,
    silence_audio_backend_warnings,
    sweep_temp_files,
)

logger = logging.getLogger(__name__)

BLOCK_SEC = 60


def pick_top_two_speakers(segments: list[Segment]) -> tuple[str, str]:
    speaker_time: dict[str, float] = defaultdict(float)
    for seg in segments:
        speaker_time[seg.speaker] += seg.duration
    ranked = sorted(speaker_time.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        raise ValueError(f"need ≥2 speakers, got {len(ranked)}")
    return ranked[0][0], ranked[1][0]


def assign_channels(mp3_name: str, speakers: tuple[str, str]) -> dict[str, int]:
    digest = hashlib.md5(mp3_name.encode()).digest()[0]
    if digest % 2 == 0:
        return {speakers[0]: 0, speakers[1]: 1}
    return {speakers[1]: 0, speakers[0]: 1}


def speaker_ranges(
    segments: list[Segment],
    channel_by_speaker: dict[str, int],
    sample_rate: int,
    n_samples: int,
) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    for seg in segments:
        channel = channel_by_speaker.get(seg.speaker)
        if channel is None:
            continue
        start = max(0, int(seg.start * sample_rate))
        end = min(n_samples, int(seg.end * sample_rate))
        if end > start:
            ranges.append((start, end, channel))
    ranges.sort()
    return ranges


def build_one(
    audio_path: Path,
    rttm_path: Path,
    out_path: Path,
    sample_rate: int,
    min_share: float,
) -> Path | None:
    if out_path.exists():
        logger.debug("skip (exists): %s", out_path)
        return out_path

    segments = parse_rttm(rttm_path)
    if not segments:
        logger.warning("empty rttm: %s", rttm_path)
        return None

    n_dominant = count_dominant_speakers(segments, min_share)
    if n_dominant != 2:
        logger.debug(
            "reject %s (%d dominant speakers)", audio_path.name, n_dominant
        )
        return None

    try:
        top_two = pick_top_two_speakers(segments)
    except ValueError as exc:
        logger.warning("skip %s: %s", audio_path.name, exc)
        return None

    channel_by_speaker = assign_channels(audio_path.name, top_two)

    with sf.SoundFile(str(audio_path)) as fin:
        if fin.samplerate != sample_rate or fin.channels != 1:
            raise ValueError(
                f"{audio_path.name} is {fin.samplerate} Hz / {fin.channels} ch, "
                f"expected {sample_rate} Hz mono; it did not come from "
                "decode_to_wav"
            )
        sr = fin.samplerate
        n_samples = len(fin)
        ranges = speaker_ranges(segments, channel_by_speaker, sr, n_samples)

        with atomic_path(out_path) as tmp:
            with sf.SoundFile(
                str(tmp),
                "w",
                samplerate=sr,
                channels=2,
                subtype="PCM_16",
                # Spelled out because atomic_path hands us a `.tmp-<pid>` name
                # and there is no extension left to infer the container from.
                format="WAV",
            ) as fout:
                # `first` is the oldest range that might still overlap the
                # block ahead; ranges before it are entirely behind us. Without
                # it every block would rescan all of an episode's segments.
                first = offset = 0
                while True:
                    block = fin.read(BLOCK_SEC * sr, dtype="float32")
                    if not len(block):
                        break
                    block_end = offset + len(block)
                    stereo = np.zeros((len(block), 2), dtype=np.float32)

                    i = first
                    while i < len(ranges) and ranges[i][0] < block_end:
                        start_sample, end_sample, channel = ranges[i]
                        start = max(start_sample, offset) - offset
                        end = min(end_sample, block_end) - offset
                        if end > start:
                            stereo[start:end, channel] = block[start:end]
                        if end_sample <= block_end and i == first:
                            first += 1
                        i += 1

                    fout.write(stereo)
                    offset = block_end

    logger.debug("wrote %s (%.1fs)", out_path, n_samples / sr)
    return out_path


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("./data/raw_mp3"),
        help="Holds the decoded wavs written by run_preprocess, not the "
        "downloaded episodes: the originals are never read here.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./data/stereo"))
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--min-share",
        type=float,
        default=0.1,
        help=(
            "A speaker is 'dominant' if accounts for at least this fraction of "
            "speech time. Must match diarize_pyannote.py."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    silence_audio_backend_warnings()
    sweep_temp_files(args.out_dir)

    audio_paths = iter_decoded_files(args.audio_dir)
    logger.info("found %d decoded episodes under %s", len(audio_paths), args.audio_dir)

    written = skipped = 0
    for audio_path in audio_paths:
        rttm_path = audio_path.with_suffix(".rttm")
        if not rttm_path.exists():
            logger.debug("no rttm for %s, skipping", audio_path.name)
            continue
        out_name = f"{audio_path.parent.name}__{audio_path.stem}.wav"
        out_path = args.out_dir / out_name
        try:
            result = build_one(
                audio_path, rttm_path, out_path, args.sample_rate, args.min_share
            )
        except Exception:
            logger.exception("failed to build stereo for %s", audio_path)
            continue
        if result is not None:
            written += 1
        else:
            skipped += 1

    logger.info(
        f"wrote {written} stereo wavs to {args.out_dir}; "
        f"{skipped} skipped (not exactly 2 dominant speakers, or unusable rttm)"
    )


if __name__ == "__main__":
    main()
