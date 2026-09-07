from __future__ import annotations

import argparse
import bisect
import json
import logging
import statistics
import sys
from pathlib import Path

import numpy as np
import sphn
import torch
import whisper_timestamped
import whisperx
from transformers import pipeline

from common import atomic_write, setup_logging
from torch_compat import trust_torch_checkpoints

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAIN_SPEAKER = "SPEAKER_MAIN"

# Trained on CIEMPIESS + HUB4-NE + CallHome ES + Common Voice. The CallHome part
# is why it beats WhisperX's default for this corpus: spontaneous conversation
# rather than prepared speech.
DEFAULT_ALIGN_MODEL = "carlosdanielhernandezmena/wav2vec2-large-xlsr-53-spanish-ep5-944h"

DEFAULT_ASR_MODEL = {
    "whisperx": "large-v3",
    "whisper-timestamped": "large-v3",
    # 2.0 is the multilingual one; the HF metadata says en/de but the card
    # reports disfluency F1 across ten languages.
    "crisper": "nyralabs/CrisperWhisper2.0_large",
}

# Backends that cannot produce their own word timestamps.
NEEDS_CTC = {"whisperx"}


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


def speech_regions(audio: np.ndarray, min_gap: float, min_speech: float) -> list[tuple[int, int]]:
    voiced = audio != 0.0
    edges = np.diff(np.concatenate(([0], voiced.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)

    regions: list[tuple[int, int]] = []
    gap = int(min_gap * SAMPLE_RATE)
    for start, end in zip(starts, ends):
        # A pause shorter than min_gap is part of the speech, not a boundary;
        # cutting on it would shred words that carry a brief silent stop.
        if regions and start - regions[-1][1] < gap:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((int(start), int(end)))
    least = int(min_speech * SAMPLE_RATE)
    return [(s, e) for s, e in regions if e - s >= least]


def compact(audio: np.ndarray, regions: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    pieces, index, elapsed = [], [], 0.0
    for start, end in regions:
        pieces.append(audio[start:end])
        duration = (end - start) / SAMPLE_RATE
        index.append((elapsed + duration, start / SAMPLE_RATE - elapsed))
        elapsed += duration
    return np.concatenate(pieces) if pieces else audio[:0], index


def restore_timeline(words: list[dict], index: list[tuple[float, float]]) -> list[dict]:
    if not index:
        return words
    bounds = [end for end, _ in index]
    for word in words:
        for key in ("start", "end"):
            value = word.get(key)
            if value is None:
                continue
            slot = min(bisect.bisect_right(bounds, float(value)), len(index) - 1)
            word[key] = float(value) + index[slot][1]
    return words


def words_to_segments(words: list[dict], max_sec: float = 20.0) -> list[dict]:
    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    segments: list[dict] = []
    current: list[dict] = []
    for word in timed:
        if current and word["end"] - current[0]["start"] > max_sec:
            segments.append(_as_segment(current))
            current = []
        current.append(word)
    if current:
        segments.append(_as_segment(current))
    return segments


def _as_segment(words: list[dict]) -> dict:
    return {
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": " ".join(str(w["word"]).strip() for w in words),
    }


def load_whisperx(args, compute_type: str):
    asr = whisperx.load_model(
        args.asr_model, args.device, compute_type=compute_type, language=args.lang
    )

    def run(audio: np.ndarray) -> tuple[list[dict], list[dict] | None]:
        result = asr.transcribe(audio, batch_size=args.batch_size, language=args.lang)
        return result.get("segments") or [], None

    return run


def load_whisper_timestamped(args, compute_type: str):
    model = whisper_timestamped.load_model(args.asr_model, device=args.device)

    def run(audio: np.ndarray) -> tuple[list[dict], list[dict] | None]:
        result = whisper_timestamped.transcribe(model, audio, language=args.lang, verbose=None)
        segments, words = [], []
        for segment in result.get("segments") or []:
            segments.append(
                {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
            )
            for word in segment.get("words") or []:
                words.append(
                    {
                        "word": word["text"],
                        "start": word["start"],
                        "end": word["end"],
                        "score": word.get("confidence"),
                    }
                )
        return segments, words

    return run


def load_crisper(args, compute_type: str):
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.asr_model,
        device=args.device,
        chunk_length_s=30,
    )

    def run(audio: np.ndarray) -> tuple[list[dict], list[dict] | None]:
        out = asr(
            {"raw": audio, "sampling_rate": SAMPLE_RATE},
            return_timestamps="word",
            generate_kwargs={"language": args.lang},
        )
        words = []
        for chunk in out.get("chunks") or []:
            start, end = chunk.get("timestamp") or (None, None)
            words.append({"word": chunk["text"], "start": start, "end": end, "score": None})
        return words_to_segments(words), words

    return run


LOADERS = {
    "whisperx": load_whisperx,
    "whisper-timestamped": load_whisper_timestamped,
    "crisper": load_crisper,
}


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


def annotate_one(path, out_path, transcribe, align_model, align_metadata, args) -> dict:
    audio = read_channel(path, args.channel)
    regions = speech_regions(audio, args.min_gap, args.min_speech)
    if not regions:
        raise RuntimeError("channel is silent end to end")
    spliced, index = compact(audio, regions)

    segments, native_words = transcribe(spliced)
    if not segments:
        raise RuntimeError("no speech found on the transcribed channel")

    if args.align == "ctc":
        aligned = whisperx.align(
            segments,
            align_model,
            align_metadata,
            spliced,
            args.device,
            return_char_alignments=False,
        )
        words = aligned.get("word_segments") or []
    else:
        words = native_words or []
    words = restore_timeline(words, index)

    alignments, dropped = to_alignments(words)
    if not alignments:
        raise RuntimeError(f"placed none of the {len(words)} word(s)")

    stats = summarize(words, dropped, args.low_score)
    stats.update(
        backend=args.backend,
        asr_model=args.asr_model,
        align=args.align,
        align_model=args.align_model if args.align == "ctc" else None,
        language=args.lang,
        audio_seconds=round(len(audio) / SAMPLE_RATE, 2),
        speech_seconds=round(len(spliced) / SAMPLE_RATE, 2),
        speech_regions=len(regions),
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
    parser.add_argument("--backend", default="whisperx", choices=sorted(LOADERS))
    parser.add_argument("--align", default=None, choices=["ctc", "native"], help="Where the word timestamps come from. Defaults to ctc for backends that produce none.")
    parser.add_argument("--asr-model", default=None, help="Defaults to the usual model for the chosen backend.")
    parser.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    parser.add_argument("--channel", type=int, default=0, help="Channel to transcribe. The text stream is the main speaker's only.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compute-type", default=None, help="ctranslate2 precision. Defaults to float16 on cuda, int8 elsewhere.")
    parser.add_argument("--min-gap", type=float, default=0.5, help="Silence shorter than this is kept as part of the speech instead of splitting it.")
    parser.add_argument("--min-speech", type=float, default=0.2, help="Speech regions shorter than this are dropped as noise.")
    parser.add_argument("--low-score", type=float, default=0.3, help="Alignment score under which a word is counted as poorly placed. Recorded, never filtered: the threshold is provisional until the distribution over the corpus is known.")
    args = parser.parse_args()

    args.asr_model = args.asr_model or DEFAULT_ASR_MODEL[args.backend]
    if args.align is None:
        args.align = "ctc" if args.backend in NEEDS_CTC else "native"
    if args.align == "native" and args.backend in NEEDS_CTC:
        parser.error(f"backend {args.backend} produces no word timestamps of its own; use --align ctc")

    args.device = resolve_device(args.device)
    if args.device == "mps" and args.backend == "whisperx":
        parser.error("ctranslate2 has no Metal backend; use --device cpu on a Mac")
    compute_type = args.compute_type or ("float16" if args.device == "cuda" else "int8")

    paths = [Path(json.loads(line)["path"]) for line in args.egs.read_text(encoding="utf-8").splitlines() if line.strip()]
    logger.info(
        "%d episode(s) | device=%s backend=%s asr=%s align=%s",
        len(paths), args.device, args.backend, args.asr_model, args.align,
    )

    # Both the pyannote VAD that whisperx runs before transcribing and the
    # aligner's checkpoint fail to load under torch 2.6's defaults.
    trust_torch_checkpoints()
    transcribe = LOADERS[args.backend](args, compute_type)

    align_model = align_metadata = None
    if args.align == "ctc":
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
            stats = annotate_one(path, out_path, transcribe, align_model, align_metadata, args)
        except Exception:
            logger.exception("[%d/%d] failed on %s", index, len(paths), path.name)
            err_path.touch()
            failed += 1
            continue
        err_path.unlink(missing_ok=True)
        done += 1
        logger.info(
            "[%d/%d] %s — %d words, %.0f%% of %.0fs is speech, score median %s, %d without timing",
            index, len(paths), path.name, stats["words"],
            100 * stats["speech_seconds"] / max(stats["audio_seconds"], 1e-9),
            stats["audio_seconds"],
            stats.get("score_median", "n/a"), stats["words_without_timing"],
        )

    logger.info("annotated %d, skipped %d, failed %d", done, skipped, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
