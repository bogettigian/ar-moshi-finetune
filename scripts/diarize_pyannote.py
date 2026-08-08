from __future__ import annotations

import argparse
import logging
import logging.config
import os
from pathlib import Path

import torch

from common import count_dominant_speakers, iter_audio_files, parse_rttm

_torch_load_orig = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_trusted

from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


def load_pipeline(model: str, device: str, hf_token: str | None) -> Pipeline:
    logger.info("loading pipeline %s on %s", model, device)
    pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    pipeline.to(torch.device(device))
    return pipeline


def diarize_one(pipeline: Pipeline, audio_path: Path) -> Path:
    rttm_path = audio_path.with_suffix(".rttm")
    if rttm_path.exists():
        logger.debug("skip (exists): %s", rttm_path)
        return rttm_path
    logger.info("diarizing %s", audio_path.name)
    diarization = pipeline(str(audio_path))
    with rttm_path.open("w") as fh:
        diarization.write_rttm(fh)
    return rttm_path


def main() -> None:
    Path("logs").mkdir(exist_ok=True)
    logging.config.fileConfig("log.ini", disable_existing_loggers=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("./data/raw_mp3"))
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min-share",
        type=float,
        default=0.1,
        help="A speaker is 'dominant' if accounts for at least this fraction of speech time.",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "no HUGGINGFACE_TOKEN / HF_TOKEN in env — pyannote download may fail"
        )

    pipeline = load_pipeline(args.model, args.device, hf_token)

    audio_paths = iter_audio_files(args.in_dir)
    logger.info("found %d audio files under %s", len(audio_paths), args.in_dir)

    kept = 0
    rejected_path = args.in_dir / "diarization_rejected.txt"
    rejected_lines: list[str] = []
    for audio_path in audio_paths:
        try:
            rttm_path = diarize_one(pipeline, audio_path)
        except Exception:
            logger.exception("failed to diarize %s", audio_path)
            continue
        n_dominant = count_dominant_speakers(parse_rttm(rttm_path), args.min_share)
        if n_dominant == 2:
            kept += 1
        else:
            logger.info(
                "reject %s (%d dominant speakers)", audio_path.name, n_dominant
            )
            rejected_lines.append(f"{audio_path}\t{n_dominant}")

    if rejected_lines:
        rejected_path.write_text("\n".join(rejected_lines) + "\n")

    logger.info(
        f"diarized {len(audio_paths)} files, {kept} kept (2 dominant speakers), "
        f"{len(rejected_lines)} rejected: {rejected_path}"
    )


if __name__ == "__main__":
    main()
