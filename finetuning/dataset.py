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
from typing import Any, Dict, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

try:
    from finetuning.language_utils import canonicalize_language_name, normalize_language_key
except ImportError:
    from language_utils import canonicalize_language_name, normalize_language_key

AudioLike = Union[
    str,
    np.ndarray,
    Tuple[np.ndarray, int],
]


class TTSDataset(Dataset):
    def __init__(
        self,
        data_list,
        processor,
        config: Qwen3TTSConfig,
        xvector_only_ratio: float = 0.2,
    ):
        self.data_list = data_list
        self.processor = processor
        self.config = config
        self.xvector_only_ratio = float(xvector_only_ratio)
        if not 0.0 <= self.xvector_only_ratio <= 1.0:
            raise ValueError(f"xvector_only_ratio must be in [0, 1], got {self.xvector_only_ratio}")

        self.codec_language_id = {
            str(key).lower(): int(value)
            for key, value in (self.config.talker_config.codec_language_id or {}).items()
        }
        self.num_code_groups = int(self.config.talker_config.num_code_groups)

    def __len__(self):
        return len(self.data_list)

    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=None, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for audio in items:
            if isinstance(audio, str):
                out.append(self._load_audio_to_np(audio))
            elif isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], np.ndarray):
                out.append((audio[0].astype(np.float32), int(audio[1])))
            elif isinstance(audio, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(audio)}")
        return out

    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    def _build_ref_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    def _tokenize_text(self, text: str) -> torch.Tensor:
        input_data = self.processor(text=text, return_tensors="pt", padding=True)
        input_ids = input_data["input_ids"]
        return input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids

    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        if sr != 24000:
            audio = librosa.resample(y=audio.astype(np.float32), orig_sr=int(sr), target_sr=24000)

        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return mels.squeeze(0)

    def _get_language_key(self, language: str) -> str:
        canonical_language = canonicalize_language_name(language)
        language_key = normalize_language_key(canonical_language)
        if language_key != "auto" and language_key not in self.codec_language_id:
            supported = sorted(self.codec_language_id.keys())
            raise ValueError(f"Unsupported language `{language}`. Supported keys: {supported}")
        return language_key

    def _get_ref_text(self, item: Dict[str, Any]) -> str:
        ref_text = item.get("ref_text", item.get("tef_text"))
        if ref_text in (None, ""):
            raise ValueError(f"`ref_text` is required for voice-clone finetuning. Bad item: {item}")
        return ref_text

    def _build_sample_tensors(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        text_full_ids = data["text_ids"][0]
        if text_full_ids.shape[0] <= 8:
            raise ValueError("Target text is too short after tokenization.")
        role_ids = text_full_ids[:3]
        target_text_ids = text_full_ids[3:-5]
        if target_text_ids.numel() == 0:
            raise ValueError("Target text became empty after removing control tokens.")

        ref_text_full_ids = data["ref_text_ids"][0]
        if ref_text_full_ids.shape[0] <= 5:
            raise ValueError("Reference text is too short after tokenization.")
        ref_text_ids = ref_text_full_ids[3:-2]

        audio_codes = data["audio_codes"]
        ref_audio_codes = data["ref_audio_codes"]
        if audio_codes.shape[-1] != self.num_code_groups or ref_audio_codes.shape[-1] != self.num_code_groups:
            raise ValueError(
                f"Expected {self.num_code_groups} code groups, got "
                f"{audio_codes.shape[-1]} and {ref_audio_codes.shape[-1]}"
            )

        language_key = data["language_key"]
        language_id = None if language_key == "auto" else self.codec_language_id[language_key]

        if language_id is None:
            codec_prefill = [
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                self.config.talker_config.codec_think_id,
                self.config.talker_config.codec_think_bos_id,
                language_id,
                self.config.talker_config.codec_think_eos_id,
            ]

        speaker_prefix_index = len(codec_prefill)
        codec_prefix = codec_prefill + [0, self.config.talker_config.codec_pad_id]
        text_prefix = [self.config.tts_pad_token_id] * (len(codec_prefix) - 1) + [self.config.tts_bos_token_id]
        speaker_position = 3 + speaker_prefix_index

        text_tokens: List[int] = []
        codec_tokens: List[int] = []
        codec_embedding_mask: List[bool] = []
        codec_frame_values: List[List[int]] = []
        all_codec_mask: List[bool] = []
        target_codec_mask: List[bool] = []
        labels: List[int] = []

        zero_codec_frame = [0] * self.num_code_groups

        def add_step(
            text_id: int,
            codec_id: int,
            *,
            codec_on: bool,
            frame_codes: torch.Tensor = None,
            mark_target: bool = False,
        ) -> None:
            text_tokens.append(int(text_id))
            codec_tokens.append(int(codec_id))
            codec_embedding_mask.append(bool(codec_on))
            if frame_codes is None:
                codec_frame_values.append(list(zero_codec_frame))
                all_codec_mask.append(False)
            else:
                codec_frame_values.append(frame_codes.to(dtype=torch.long).tolist())
                all_codec_mask.append(True)
            target_codec_mask.append(bool(mark_target))
            labels.append(int(frame_codes[0].item()) if mark_target else -100)

        for role_id in role_ids.tolist():
            add_step(role_id, 0, codec_on=False)

        for index, (text_id, codec_id) in enumerate(zip(text_prefix, codec_prefix)):
            add_step(text_id, codec_id, codec_on=(index != speaker_prefix_index))

        if data["use_xvector_only"]:
            for token_id in target_text_ids.tolist():
                add_step(token_id, self.config.talker_config.codec_pad_id, codec_on=True)
            add_step(self.config.tts_eos_token_id, self.config.talker_config.codec_pad_id, codec_on=True)
            add_step(self.config.tts_pad_token_id, self.config.talker_config.codec_bos_id, codec_on=True)

            for frame_codes in audio_codes:
                add_step(
                    self.config.tts_pad_token_id,
                    int(frame_codes[0].item()),
                    codec_on=True,
                    frame_codes=frame_codes,
                    mark_target=True,
                )
        else:
            if ref_text_ids.numel() == 0:
                raise ValueError("ICL training sample requires non-empty ref_text.")

            for token_id in ref_text_ids.tolist():
                add_step(token_id, self.config.talker_config.codec_pad_id, codec_on=True)
            for token_id in target_text_ids.tolist():
                add_step(token_id, self.config.talker_config.codec_pad_id, codec_on=True)
            add_step(self.config.tts_eos_token_id, self.config.talker_config.codec_pad_id, codec_on=True)
            add_step(self.config.tts_pad_token_id, self.config.talker_config.codec_bos_id, codec_on=True)

            for frame_codes in ref_audio_codes:
                add_step(
                    self.config.tts_pad_token_id,
                    int(frame_codes[0].item()),
                    codec_on=True,
                    frame_codes=frame_codes,
                    mark_target=False,
                )

            for frame_codes in audio_codes:
                add_step(
                    self.config.tts_pad_token_id,
                    int(frame_codes[0].item()),
                    codec_on=True,
                    frame_codes=frame_codes,
                    mark_target=True,
                )

        add_step(self.config.tts_pad_token_id, self.config.talker_config.codec_eos_token_id, codec_on=True)
        labels[-1] = self.config.talker_config.codec_eos_token_id

        seq_len = len(text_tokens)
        input_ids = torch.zeros((seq_len, 2), dtype=torch.long)
        input_ids[:, 0] = torch.tensor(text_tokens, dtype=torch.long)
        input_ids[:, 1] = torch.tensor(codec_tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "codec_ids": torch.tensor(codec_frame_values, dtype=torch.long),
            "codec_embedding_mask": torch.tensor(codec_embedding_mask, dtype=torch.bool),
            "all_codec_mask": torch.tensor(all_codec_mask, dtype=torch.bool),
            "target_codec_mask": torch.tensor(target_codec_mask, dtype=torch.bool),
            "codec_0_labels": torch.tensor(labels, dtype=torch.long),
            "speaker_position": torch.tensor(speaker_position, dtype=torch.long),
        }

    def __getitem__(self, idx):
        item = self.data_list[idx]

        language_key = self._get_language_key(item["language"])
        text_ids = self._tokenize_text(self._build_assistant_text(item["text"]))
        ref_text_ids = self._tokenize_text(self._build_ref_text(self._get_ref_text(item)))

        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        ref_audio_codes = torch.tensor(item["ref_audio_codes"], dtype=torch.long)

        ref_audio_list = self._normalize_audio_inputs(item["ref_audio"])
        wav, sr = ref_audio_list[0]
        ref_mel = self.extract_mels(audio=wav, sr=sr)

        use_xvector_only = torch.rand(1).item() < self.xvector_only_ratio

        return {
            "text_ids": text_ids,
            "ref_text_ids": ref_text_ids,
            "audio_codes": audio_codes,
            "ref_audio_codes": ref_audio_codes,
            "ref_mel": ref_mel,
            "language_key": language_key,
            "use_xvector_only": use_xvector_only,
        }

    def collate_fn(self, batch):
        built_batch = [self._build_sample_tensors(sample) for sample in batch]

        batch_size = len(built_batch)
        max_length = max(item["input_ids"].shape[0] for item in built_batch)
        mel_dim = int(batch[0]["ref_mel"].shape[-1])
        max_ref_mel_length = max(sample["ref_mel"].shape[0] for sample in batch)

        input_ids = torch.zeros((batch_size, max_length, 2), dtype=torch.long)
        codec_ids = torch.zeros((batch_size, max_length, self.num_code_groups), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        text_embedding_mask = torch.zeros((batch_size, max_length, 1), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((batch_size, max_length, 1), dtype=torch.bool)
        codec_0_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        all_codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        target_codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        speaker_positions = torch.zeros((batch_size,), dtype=torch.long)
        ref_mels = torch.zeros((batch_size, max_ref_mel_length, mel_dim), dtype=torch.float32)

        for index, (sample, built) in enumerate(zip(batch, built_batch)):
            seq_len = built["input_ids"].shape[0]
            input_ids[index, :seq_len] = built["input_ids"]
            codec_ids[index, :seq_len] = built["codec_ids"]
            attention_mask[index, :seq_len] = 1
            text_embedding_mask[index, :seq_len, 0] = True
            codec_embedding_mask[index, :seq_len, 0] = built["codec_embedding_mask"]
            codec_0_labels[index, :seq_len] = built["codec_0_labels"]
            all_codec_mask[index, :seq_len] = built["all_codec_mask"]
            target_codec_mask[index, :seq_len] = built["target_codec_mask"]
            speaker_positions[index] = built["speaker_position"]

            ref_mel = sample["ref_mel"]
            ref_mels[index, :ref_mel.shape[0]] = ref_mel

        return {
            "input_ids": input_ids,
            "ref_mels": ref_mels,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask,
            "codec_embedding_mask": codec_embedding_mask,
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "all_codec_mask": all_codec_mask,
            "codec_mask": target_codec_mask,
            "speaker_positions": speaker_positions,
        }
