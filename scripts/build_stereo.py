from __future__ import annotations

import argparse
import hashlib
import logging
import logging.config
from collections import defaultdict
from pathlib import Path

import numpy as np
import sphn
import torchaudio
import torchaudio.functional as F

from common import Segment, count_dominant_speakers, iter_audio_files, parse_rttm

logger = logging.getLogger(__name__)


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
        logger.info(
            "reject %s (%d dominant speakers)", audio_path.name, n_dominant
        )
        return None

    try:
        top_two = pick_top_two_speakers(segments)
    except ValueError as exc:
        logger.warning("skip %s: %s", audio_path.name, exc)
        return None

    channel_by_speaker = assign_channels(audio_path.name, top_two)
    keep_speakers = set(top_two)

    waveform, sr_orig = torchaudio.load(str(audio_path))
    if sr_orig != sample_rate:
        waveform = F.resample(waveform, sr_orig, sample_rate)
    sr = sample_rate
    arr = waveform.numpy().astype(np.float32)
    mono = arr[0] if arr.ndim > 1 else arr
    n_samples = mono.shape[-1]

    stereo = np.zeros((2, n_samples), dtype=np.float32)
    for seg in segments:
        if seg.speaker not in keep_speakers:
            continue
        ch = channel_by_speaker[seg.speaker]
        start_sample = max(0, int(seg.start * sr))
        end_sample = min(n_samples, int(seg.end * sr))
        if end_sample <= start_sample:
            continue
        stereo[ch, start_sample:end_sample] = mono[start_sample:end_sample]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(out_path), stereo, sample_rate=sr)
    logger.info("wrote %s (%.1fs)", out_path, n_samples / sr)
    return out_path


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("./data/raw_mp3"))
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

    audio_paths = iter_audio_files(args.audio_dir)
    logger.info("found %d audio files under %s", len(audio_paths), args.audio_dir)

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
