"""Run the released English word-reader on one WAV/FLAC file shorter than 30 seconds."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "reader" / "train"
sys.path.insert(0, str(TRAIN))

from cache_soft_tokens import load_encoder, soft_tokens  # noqa: E402
from decode import load_reader_dir, transcribe_batch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--encoder", default="google/embeddinggemma-2")
    ap.add_argument("--reader-dir", type=Path, default=ROOT / "reader")
    args = ap.parse_args()

    wav, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1 or not len(wav):
        raise ValueError("Expected a non-empty mono or stereo audio file")
    if sr != 16_000:
        g = math.gcd(int(sr), 16_000)
        wav = resample_poly(wav, 16_000 // g, int(sr) // g).astype(np.float32)
    if len(wav) > 30 * 16_000:
        raise ValueError("This quick test accepts clips up to 30 seconds")

    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    load_start = time.perf_counter()
    fe, encoder, _ = load_encoder(args.encoder, args.device)
    reader, table, vocab, _ = load_reader_dir(str(args.reader_dir), args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    load_seconds = time.perf_counter() - load_start

    transcribe_start = time.perf_counter()
    tokens, _ = soft_tokens(encoder, fe, wav, args.device)
    text = transcribe_batch(reader, [tokens], table, vocab, args.device)[0]
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    transcribe_seconds = time.perf_counter() - transcribe_start

    print(text)
    print(f"Model load: {load_seconds:.2f}s | Transcription: {transcribe_seconds:.2f}s")


if __name__ == "__main__":
    main()
