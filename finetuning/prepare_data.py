# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import time
from typing import Dict, List

from qwen_tts import Qwen3TTSTokenizer

try:
    from finetuning.language_utils import canonicalize_language_name
except ImportError:
    from language_utils import canonicalize_language_name

BATCH_INFER_NUM = 32


def _normalize_record(line: Dict) -> Dict:
    if "ref_text" not in line and "tef_text" in line:
        line["ref_text"] = line.pop("tef_text")

    required_fields = ("audio", "text", "ref_audio", "ref_text", "language")
    missing = [field for field in required_fields if field not in line or line[field] in (None, "")]
    if missing:
        raise ValueError(f"Missing required fields {missing} in record: {line}")

    line["language"] = canonicalize_language_name(line["language"])
    return line


def _encode_missing_codes(
    tokenizer_12hz: Qwen3TTSTokenizer,
    records: List[Dict],
    source_field: str,
    output_field: str,
) -> None:
    cache: Dict[str, List[List[int]]] = {}
    pending_inputs: List[str] = []

    for line in records:
        if line.get(output_field) is not None:
            continue

        audio_ref = line[source_field]
        if not isinstance(audio_ref, str):
            raise TypeError(
                f"`{source_field}` must be a path/URL/base64 string in prepare_data, got {type(audio_ref)}"
            )

        if audio_ref not in cache:
            pending_inputs.append(audio_ref)
            cache[audio_ref] = None

    total_pending = len(pending_inputs)
    if total_pending == 0:
        print(f"[prepare_data] {output_field}: no missing items, skipping encoding")
        return

    print(
        f"[prepare_data] {output_field}: encoding {total_pending} unique audio files "
        f"in batches of {BATCH_INFER_NUM}"
    )
    started_at = time.time()

    for start in range(0, len(pending_inputs), BATCH_INFER_NUM):
        batch_inputs = pending_inputs[start:start + BATCH_INFER_NUM]
        if not batch_inputs:
            continue

        enc_res = tokenizer_12hz.encode(batch_inputs)
        for audio_ref, code in zip(batch_inputs, enc_res.audio_codes):
            cache[audio_ref] = code.cpu().tolist()

        processed = min(start + len(batch_inputs), total_pending)
        elapsed = time.time() - started_at
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(
            f"[prepare_data] {output_field}: "
            f"{processed}/{total_pending} encoded "
            f"({processed / total_pending * 100:.1f}%), "
            f"{rate:.2f} files/s"
        )

    for line in records:
        if line.get(output_field) is None:
            line[output_field] = cache[line[source_field]]


def main():
    overall_started_at = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    args = parser.parse_args()

    print(f"[prepare_data] loading tokenizer from {args.tokenizer_model_path} on {args.device}")

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    print(f"[prepare_data] reading input JSONL: {args.input_jsonl}")

    total_lines = open(args.input_jsonl).readlines()
    total_lines = [_normalize_record(json.loads(line.strip())) for line in total_lines]

    missing_audio_codes = sum(1 for line in total_lines if line.get("audio_codes") is None)
    missing_ref_audio_codes = sum(1 for line in total_lines if line.get("ref_audio_codes") is None)
    print(
        f"[prepare_data] loaded {len(total_lines)} rows | "
        f"missing audio_codes: {missing_audio_codes} | "
        f"missing ref_audio_codes: {missing_ref_audio_codes}"
    )

    _encode_missing_codes(tokenizer_12hz, total_lines, source_field="audio", output_field="audio_codes")
    _encode_missing_codes(tokenizer_12hz, total_lines, source_field="ref_audio", output_field="ref_audio_codes")

    final_lines = [json.dumps(line, ensure_ascii=False) for line in total_lines]

    print(f"[prepare_data] writing output JSONL: {args.output_jsonl}")
    with open(args.output_jsonl, 'w') as f:
        for line in final_lines:
            f.writelines(line + '\n')

    elapsed = time.time() - overall_started_at
    print(
        f"[prepare_data] done | rows: {len(total_lines)} | "
        f"elapsed: {elapsed:.1f}s | output: {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
