from __future__ import annotations
import argparse
import time
import imageio_ffmpeg
import whisper
import whisper.audio

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with OpenAI Whisper.")
    parser.add_argument("audio")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    whisper.audio.FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    start = time.perf_counter()
    model = whisper.load_model(args.model, device=args.device)
    loaded = time.perf_counter()
    result = model.transcribe(args.audio, language="en", fp16=args.device.startswith("cuda"))
    done = time.perf_counter()
    print(result["text"].strip())
    print(f"Model load: {loaded - start:.2f}s | Transcription: {done - loaded:.2f}s")

if __name__ == "__main__":
    main()
