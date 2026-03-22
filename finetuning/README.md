## Fine-Tuning Qwen3-TTS-12Hz Base For Multilingual Voice Clone

This workflow keeps the checkpoint as a `base` model, so the result stays compatible with `generate_voice_clone(...)` and `create_voice_clone_prompt(...)`.

It is designed for adding a new language to the existing multilingual Base model, while keeping the original voice-clone path intact. The same script works with both `Qwen3-TTS-12Hz-1.7B-Base` and `Qwen3-TTS-12Hz-0.6B-Base`.

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

### 1) Input JSONL format

Prepare a JSONL file where each line contains:

- `audio`: target training audio path
- `text`: transcript for `audio`
- `ref_audio`: reference audio path used for voice cloning
- `ref_text`: transcript of `ref_audio`
- `language`: canonical language name, for example `Hungarian`, `English`, `German`

Example:

```jsonl
{"audio":"./data/utt0001.wav","text":"Ez egy magyar mondat.","ref_audio":"./data/ref.wav","ref_text":"Ez a referenciahang szövege.","language":"Hungarian"}
{"audio":"./data/utt0002.wav","text":"This line keeps a bit of English in the mix.","ref_audio":"./data/ref.wav","ref_text":"This is the reference transcript.","language":"English"}
```

Notes:

- `language` is required on every row.
- The scripts accept `tef_text` as a legacy typo alias, but the canonical field name is `ref_text`.
- Reusing the same `ref_audio` across many samples is usually the most stable setup.

### 2) Prepare data

`prepare_data.py` extracts:

- `audio_codes`: 12 Hz tokenizer codes for the target `audio`
- `ref_audio_codes`: 12 Hz tokenizer codes for the reference `ref_audio`

These fields are stored as JSON lists with shape `T x 16`, where `T` is the codec length in frames and `16` is the number of code groups used by the 12 Hz tokenizer.

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

If `audio_codes` or `ref_audio_codes` already exist in the input JSONL, they are kept and only the missing side is generated.

### 3) Fine-tune

The script:

- keeps `tts_model_type=base`
- registers the new language in `talker_config.codec_language_id`
- initializes the new language embedding from an existing language
- mixes ICL and x-vector-only training, with ICL kept dominant by default

Hungarian example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m finetuning.sft_12hz \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path ./output_hungarian_1p7b \
  --train_jsonl ./train_with_codes.jsonl \
  --batch_size 1 \
  --lr 2e-6 \
  --num_epochs 3 \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --new_language Hungarian \
  --new_language_init_from German \
  --xvector_only_ratio 0.2 \
  --save_steps 200 \
  --resume_from_checkpoint ./output_hungarian_1p7b/checkpoint-step-100000
```

Key arguments:

- `--new_language`: new language name to register, for example `Hungarian`
- `--new_language_init_from`: existing language used to initialize the new language embedding, for example `German`
- `--new_language_codec_id`: optional manual token id; if omitted, the next free condition token id is used
- `--xvector_only_ratio`: fraction of batches trained in `x_vector_only` mode; `0.0` means ICL-only training
- `--save_steps`: optional step-based checkpoint interval; `0` disables intermediate saves
- `--resume_from_checkpoint`: optional path to a previously saved checkpoint directory for full trainer-state resume

Checkpoints are written to:

- `output_hungarian_1p7b/checkpoint-step-200`
- `output_hungarian_1p7b/checkpoint-step-400`
- `output_hungarian_1p7b/checkpoint-epoch-0`

### 4) Quick inference test

The finetuning script builds prompts in non-streaming style, so `non_streaming_mode=True` is the recommended inference setting for the resulting checkpoint.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_voice_clone(
    text="A finetuned modell most mar tud magyarul beszelni.",
    language="Hungarian",
    ref_audio="./data/ref.wav",
    ref_text="Ez a referenciahang szovege.",
    non_streaming_mode=True,
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
NEW_LANGUAGE="Hungarian"
INIT_FROM_LANGUAGE="German"
XVECTOR_ONLY_RATIO=0.2

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

CUDA_VISIBLE_DEVICES=0 python -m finetuning.sft_12hz \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --gradient_accumulation_steps 4 \
  --mixed_precision bf16 \
  --new_language ${NEW_LANGUAGE} \
  --new_language_init_from ${INIT_FROM_LANGUAGE} \
  --xvector_only_ratio ${XVECTOR_ONLY_RATIO} \
  --save_steps 200 \
  --resume_from_checkpoint ./output_hungarian_1p7b/checkpoint-step-100000
```
