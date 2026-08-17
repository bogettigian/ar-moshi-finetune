from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
import sphn
import torch
import whisperx

from common import atomic_write, setup_logging
from torch_compat import trust_torch_checkpoints

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAIN_SPEAKER = "SPEAKER_MAIN"
# Trained on CIEMPIESS + HUB4-NE + CallHome ES + Common Voice. The CallHome part
# is why it beats WhisperX's default for this corpus: spontaneous conversation
# rather than prepared speech.
DEFAULT_ALIGN_MODEL = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"


def resolve_device(preferred: str | None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    # ctranslate2 has no Metal backend, so a Mac runs this on the CPU. Fine for
    # exercising the code path, useless for the corpus.
    return "cpu"


def read_channel(path: Path, channel: int) -> np.ndarray:
    audio, _ = sphn.read(str(path), sample_rate=SAMPLE_RATE)
    if channel >= audio.shape[0]:
        raise ValueError(f"{path.name} has {audio.shape[0]} channel(s), asked for {channel}")
    return np.ascontiguousarray(audio[channel], dtype=np.float32)


def to_alignments(words: list[dict]) -> tuple[list, int]:
    alignments = []
    dropped = 0
    for word in words:
        text = str(word.get("word", "")).strip()
        start, end = word.get("start"), word.get("end")
        if not text or start is None or end is None:
            dropped += 1
            continue
        alignments.append([text, [float(start), float(end)], MAIN_SPEAKER])
    return alignments, dropped


def summarize(words: list[dict], dropped: int, low_score: float) -> dict:
    scores = [float(w["score"]) for w in words if w.get("score") is not None]
    stats = {
        "words": len(words),
        "words_without_timing": dropped,
        "words_below_threshold": sum(1 for s in scores if s < low_score),
        "score_threshold": low_score,
    }
    if scores:
        ordered = sorted(scores)
        stats["score_mean"] = round(statistics.fmean(scores), 4)
        stats["score_median"] = round(statistics.median(ordered), 4)
        stats["score_p10"] = round(ordered[len(ordered) // 10], 4)
        stats["score_min"] = round(ordered[0], 4)
    return stats


def annotate_one(
    path: Path,
    out_path: Path,
    asr,
    align_model,
    align_metadata,
    args: argparse.Namespace,
) -> dict:
    audio = read_channel(path, args.channel)
    transcription = asr.transcribe(audio, batch_size=args.batch_size, language=args.lang)
    segments = transcription.get("segments") or []
    if not segments:
        raise RuntimeError("no speech found on the transcribed channel")

    aligned = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )
    words = aligned.get("word_segments") or []
    alignments, dropped = to_alignments(words)
    if not alignments:
        raise RuntimeError(f"alignment placed none of the {len(words)} word(s)")

    stats = summarize(words, dropped, args.low_score)
    stats.update(
        asr_model=args.whisper_model,
        align_model=args.align_model,
        language=args.lang,
        audio_seconds=round(len(audio) / SAMPLE_RATE, 2),
    )
    # The interleaver reads `alignments` and `text_conditions` and ignores every
    # other key, so the quality record travels with the data instead of in a
    # sidecar that could drift away from it.
    with atomic_write(out_path) as fh:
        json.dump({"alignments": alignments, "alignment_stats": stats}, fh, ensure_ascii=False)
    return stats


def main() -> int:
    setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("egs", type=Path, help="Manifest of wavs to annotate.")
    parser.add_argument("--lang", default="es")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    parser.add_argument("--channel", type=int, default=0, help="Channel to transcribe. The text stream is the main speaker's only.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None, help="ctranslate2 precision. Defaults to float16 on cuda, int8 elsewhere.")
    parser.add_argument("--low-score", type=float, default=0.3, help="Alignment score under which a word is counted as poorly placed. Recorded, never filtered: the threshold is provisional until the distribution over the corpus is known.")
    args = parser.parse_args()

    args.device = resolve_device(args.device)
    if args.device == "mps":
        parser.error("ctranslate2 has no Metal backend; use --device cpu on a Mac")
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")

    paths = [Path(json.loads(line)["path"]) for line in args.egs.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(
        "%d episode(s) | device=%s compute=%s asr=%s align=%s",
        len(paths), args.device, compute_type, args.whisper_model, args.align_model,
    )

    # whisperx runs a pyannote VAD before transcribing, and that checkpoint does
    # not load under torch 2.6's defaults.
    trust_torch_checkpoints()
    asr = whisperx.load_model(
        args.whisper_model, args.device, compute_type=compute_type, language=args.lang
    )
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.lang, device=args.device, model_name=args.align_model
    )

    done = failed = skipped = 0
    for index, path in enumerate(paths, 1):
        out_path = path.with_suffix(".json")
        err_path = path.with_suffix(".json.err")
        if out_path.exists():
            skipped += 1
            continue
        try:
            stats = annotate_one(path, out_path, asr, align_model, align_metadata, args)
        except Exception:
            logger.exception("[%d/%d] failed on %s", index, len(paths), path.name)
            err_path.touch()
            failed += 1
            continue
        err_path.unlink(missing_ok=True)
        done += 1
        logger.info(
            "[%d/%d] %s — %d words, score median %s, %d without timing",
            index, len(paths), path.name, stats["words"],
            stats.get("score_median", "n/a"), stats["words_without_timing"],
        )

    logger.info("annotated %d, skipped %d, failed %d", done, skipped, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
