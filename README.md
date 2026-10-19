# English ASR inference comparison

Small Windows test bundle for comparing the released English Reader v2 with OpenAI Whisper `small` on two recordings.

## Setup (PowerShell)

From this folder, create the environment and install the packages:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
```

Run either model on either WAV file:

```powershell
uv run python transcribe_reader.py Recording.wav
uv run python transcribe_reader.py Recording2.wav

uv run python transcribe_whisper.py Recording.wav
uv run python transcribe_whisper.py Recording2.wav
```

The Reader command prints its transcript followed by model-load and transcription times. The Whisper command does the same. Model load is repeated each time a command is started. Transcription time includes audio encoding and decoding for Reader, and decoding for Whisper.

## Models and weights

- **English Reader:** reader v2 from `mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader`; `reader/reader.pt`, `reader/word_table.npy`, `reader/vocab.txt`, and `reader/reader_config.json` are included. Its inference helpers are under `reader/train/`.
- **Reader audio encoder:** `google/embeddinggemma-2`. It is downloaded from Hugging Face and cached outside this folder on first use.
- **Whisper:** OpenAI Whisper `small`, installed as a Python package. Its checkpoint is downloaded on first use to the user's Whisper cache (normally `%USERPROFILE%\.cache\whisper`); it is not included in this project and is ignored by `.gitignore` if copied into the project.

The two included Reader weight files, `reader/reader.pt` and `reader/word_table.npy`, are stored with Git LFS because they exceed GitHub's regular file-size limit. Install Git LFS before cloning this repository so Git can fetch their contents.

## Background and provenance

### Shahir's idea

This experiment follows the idea credited to Shahir (`@TslShahir`) and the [Embedding Gemma Transcribe app](https://eg2.haku.sh/): keep Google's EmbeddingGemma 2 audio encoder and train a small reader to map its per-time-step audio representations to words in the model's text embedding space. The upstream [English Reader reproduction](https://github.com/john-rocky/eg2-transcribe) describes itself as an independent implementation; Shahir's app code and weights are not the files used here.

### Training code and English Reader weights

The training recipe comes from [`john-rocky/eg2-transcribe/reader/train`](https://github.com/john-rocky/eg2-transcribe/tree/main/reader/train). Its training-related Python files include `cache_soft_tokens.py`, `data.py`, `make_table.py`, `model.py`, `train.py`, `eval_wer.py`, and `decode.py`. This small inference bundle includes only `cache_soft_tokens.py`, `model.py`, and `decode.py`, which `transcribe_reader.py` needs; it does not include the training script or training data.

The Reader v2 checkpoint, vocabulary, word table, and configuration are published at [mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader on Hugging Face](https://huggingface.co/mlboydaisuke/EmbeddingGemma-2-Transcribe-Reader). The upstream release notes state that the reader was trained with the recipe above on LibriSpeech train-clean-100, train-clean-360, and train-other-500 (about 890 hours retained after out-of-vocabulary filtering). `reader.pt` is the trained Reader; `word_table.npy` contains the word embeddings made from EmbeddingGemma 2 and the LibriSpeech lexicon. The encoder itself is Google's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), fetched separately on first inference.

### Related Norwegian and Vietnamese projects

- [`ahdrage/norsk-asr`](https://github.com/ahdrage/norsk-asr) is a separate Norwegian Bokmål ASR project using the EmbeddingGemma 2 audio encoder. Its README says its recipe follows [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0).
- [QUAN-Transcribe](https://github.com/qnguyen3/QUAN-Transcribe) and its [Vietnamese model weights](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0) are Quan Nguyen's Vietnamese ASR project. It is a related architectural reference for `norsk-asr`; its code and weights are not the English Reader files in this folder.

## Recordings and results

Times below are the values printed by the user's latest runs on an NVIDIA GeForce RTX 5060 Ti with CUDA. Model-load time is per command; transcription time is per clip.

| Recording | Model | Model load | Transcription | WER |
|---|---|---:|---:|---:|
| `Recording.wav` | English Reader v2 | 5.43 s | 0.68 s | 18.37% (9/49 words) |
| `Recording.wav` | Whisper small | 1.63 s | 1.31 s | 8.16% (4/49 words) |
| `Recording2.wav` | English Reader v2 | 5.44 s | 0.61 s | 22.83% (21/92 words) |
| `Recording2.wav` | Whisper small | 1.51 s | 1.77 s | 10.87% (10/92 words) |
| **Combined** | **English Reader v2** | — | — | **21.28% (30/141 words)** |
| **Combined** | **Whisper small** | — | — | **9.93% (14/141 words)** |

WER was scored against the two test passages provided in the chat. Scoring ignores capitalization and punctuation, and normalizes `500` to “five hundred” and `14th` to “fourteenth”. Keep in mind these results are specific to these two recordings and the supplied reference passages.

### Reference text

**Recording.wav**

> Hi, my name is Mani Teja. I'm a machine learning engineer at Vanguard India. I work on building and improving machine learning systems. I enjoy solving practical problems, learning new things and turning ideas into useful tools. Today I'm recording this passage to test an English speech recognition model.

**Recording2.wav**

> Hello, I'm a Vanguard customer, and I'm calling because I have a complaint about a recent transaction in my account. I noticed a withdrawal of five hundred dollars on October fourteenth, but I don't recognize it. I checked my recent activity and confirmed that I did not make or authorize this transaction. Please investigate it, explain what happened, and let me know whether the money can be returned. My name is Mani Teja. You can contact me using the phone number and email address on my account. Thank you for your help.

## Project structure

```text
asr-inference/
├── .gitattributes
├── .gitignore
├── Recording.wav
├── Recording2.wav
├── README.md
├── requirements.txt
├── transcribe_reader.py
├── transcribe_whisper.py
└── reader/
    ├── reader.pt
    ├── reader_config.json
    ├── vocab.txt
    ├── word_table.npy
    └── train/
        ├── cache_soft_tokens.py
        ├── decode.py
        └── model.py
```
