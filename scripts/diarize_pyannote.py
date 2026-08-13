from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from common import (
    atomic_write,
    count_dominant_speakers,
    iter_decoded_files,
    parse_rttm,
    setup_logging,
    silence_audio_backend_warnings,
    sweep_temp_files,
)

_torch_load_orig = torch.load


def _torch_load_trusted(*args, **kwargs):
    kwargs["weights_only"] = False
    return _torch_load_orig(*args, **kwargs)


torch.load = _torch_load_trusted

from pyannote.audio import Pipeline

logger = logging.getLogger(__name__)


def load_pipeline(
    model: str, device: str, hf_token: str | None, batch_size: int = 32
) -> Pipeline:
    logger.info("loading pipeline %s on %s (batch size %d)", model, device, batch_size)
    pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    pipeline.to(torch.device(device))

    pipeline.segmentation_batch_size = batch_size
    pipeline.embedding_batch_size = 1
    return pipeline


def diarize_one(pipeline: Pipeline, audio_path: Path) -> Path:
    rttm_path = audio_path.with_suffix(".rttm")
    if rttm_path.exists():
        logger.debug("skip (exists): %s", rttm_path)
        return rttm_path
    logger.info("diarizing %s", audio_path.name)
    diarization = pipeline(str(audio_path))
    with atomic_write(rttm_path) as fh:
        diarization.write_rttm(fh)
    return rttm_path


def resolve_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("./data/raw_mp3"))
    parser.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Windows per forward pass. pyannote defaults to 1, which leaves a "
        "GPU idle; lower this when running on cpu (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Default: cuda, else mps, else cpu.",
    )
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

    silence_audio_backend_warnings()
    sweep_temp_files(args.in_dir)
    pipeline = load_pipeline(
        args.model, resolve_device(args.device), hf_token, args.batch_size
    )

    audio_paths = iter_decoded_files(args.in_dir)
    logger.info(
        "found %d decoded episodes under %s", len(audio_paths), args.in_dir
    )

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
        rejected_path.write_text(
            "\n".join(rejected_lines) + "\n", encoding="utf-8"
        )

    logger.info(
        f"diarized {len(audio_paths)} files, {kept} kept (2 dominant speakers), "
        f"{len(rejected_lines)} rejected: {rejected_path}"
    )


if __name__ == "__main__":
    main()
