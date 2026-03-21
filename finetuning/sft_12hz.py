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
import os
from typing import Dict

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

try:
    from finetuning.dataset import TTSDataset
    from finetuning.language_utils import canonicalize_language_name, normalize_language_key
except ImportError:
    from dataset import TTSDataset
    from language_utils import canonicalize_language_name, normalize_language_key


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--xvector_only_ratio", type=float, default=0.2)
    parser.add_argument("--sub_talker_loss_weight", type=float, default=0.3)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--new_language", type=str, required=True)
    parser.add_argument("--new_language_init_from", type=str, default="German")
    parser.add_argument("--new_language_codec_id", type=int, default=None)
    return parser.parse_args()


def _build_supported_languages(codec_language_id: Dict[str, int]):
    supported_languages = ["auto"]
    for language_key in codec_language_id.keys():
        if "dialect" not in language_key:
            supported_languages.append(language_key)
    return supported_languages


def _choose_condition_token_id(config, requested_id=None):
    used_ids = set((config.talker_config.codec_language_id or {}).values())
    used_ids.update((config.talker_config.spk_id or {}).values())
    used_ids.update(
        {
            config.talker_config.codec_pad_id,
            config.talker_config.codec_bos_id,
            config.talker_config.codec_eos_token_id,
            config.talker_config.codec_think_id,
            config.talker_config.codec_nothink_id,
            config.talker_config.codec_think_bos_id,
            config.talker_config.codec_think_eos_id,
        }
    )

    condition_id_floor = int(config.talker_config.code_predictor_config.vocab_size)
    vocab_size = int(config.talker_config.vocab_size)

    if requested_id is not None:
        if requested_id < condition_id_floor or requested_id >= vocab_size:
            raise ValueError(
                f"new_language_codec_id must be in [{condition_id_floor}, {vocab_size}), got {requested_id}"
            )
        if requested_id in used_ids:
            raise ValueError(f"new_language_codec_id={requested_id} collides with an existing reserved token id.")
        return requested_id

    for candidate in range(condition_id_floor, vocab_size):
        if candidate not in used_ids:
            return candidate

    raise ValueError("Could not find a free codec token id for the new language.")


def _register_new_language(model, language_name: str, init_from_language: str, requested_codec_id=None):
    new_language_key = normalize_language_key(canonicalize_language_name(language_name))
    init_from_key = normalize_language_key(canonicalize_language_name(init_from_language))

    codec_language_id = dict(model.config.talker_config.codec_language_id or {})
    if init_from_key not in codec_language_id:
        raise ValueError(
            f"Initialization language `{init_from_language}` is not supported by the base model. "
            f"Supported: {sorted(codec_language_id.keys())}"
        )

    if new_language_key in codec_language_id:
        if requested_codec_id is not None and codec_language_id[new_language_key] != requested_codec_id:
            raise ValueError(
                f"Language `{language_name}` already exists with codec id {codec_language_id[new_language_key]}, "
                f"but requested id was {requested_codec_id}."
            )
        new_language_id = int(codec_language_id[new_language_key])
    else:
        new_language_id = _choose_condition_token_id(model.config, requested_id=requested_codec_id)
        codec_language_id[new_language_key] = new_language_id

        init_from_id = int(codec_language_id[init_from_key])
        with torch.no_grad():
            model.talker.model.codec_embedding.weight[new_language_id].copy_(
                model.talker.model.codec_embedding.weight[init_from_id]
            )
            model.talker.codec_head.weight[new_language_id].copy_(
                model.talker.codec_head.weight[init_from_id]
            )

    model.config.talker_config.codec_language_id = codec_language_id
    model.supported_languages = _build_supported_languages(codec_language_id)

    return new_language_key, new_language_id


def _save_checkpoint(accelerator, model, processor, output_model_path, epoch):
    accelerator.wait_for_everyone()
    state_dict = accelerator.get_state_dict(model)
    if not accelerator.is_main_process:
        return

    output_dir = os.path.join(output_model_path, f"checkpoint-epoch-{epoch}")
    os.makedirs(output_dir, exist_ok=True)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.config.tts_model_type = "base"
    unwrapped_model.supported_languages = _build_supported_languages(
        unwrapped_model.config.talker_config.codec_language_id
    )
    unwrapped_model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=True,
        save_function=accelerator.save,
    )
    processor.save_pretrained(output_dir)


def train():
    args = parse_args()
    os.makedirs(args.output_model_path, exist_ok=True)
    project_config = ProjectConfiguration(
        project_dir=args.output_model_path,
        logging_dir=args.output_model_path,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if qwen3tts.model.tts_model_type != "base":
        raise ValueError(
            f"`{args.init_model_path}` is not a Base checkpoint. "
            f"Expected `tts_model_type=base`, got `{qwen3tts.model.tts_model_type}`."
        )
    if qwen3tts.model.speaker_encoder is None:
        raise ValueError("The loaded checkpoint does not expose a speaker encoder, so voice-clone finetuning is not possible.")

    new_language_name = canonicalize_language_name(args.new_language)
    init_from_name = canonicalize_language_name(args.new_language_init_from)
    new_language_key, new_language_id = _register_new_language(
        qwen3tts.model,
        language_name=new_language_name,
        init_from_language=init_from_name,
        requested_codec_id=args.new_language_codec_id,
    )

    if accelerator.is_main_process:
        accelerator.print(
            f"Registered language `{new_language_name}` as key `{new_language_key}` with codec id {new_language_id}. "
            f"Initialized from `{init_from_name}`."
        )

    qwen3tts.model.speaker_encoder.requires_grad_(False)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(
        train_data,
        qwen3tts.processor,
        qwen3tts.model.config,
        xvector_only_ratio=args.xvector_only_ratio,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(
        [parameter for parameter in qwen3tts.model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                all_codec_mask = batch["all_codec_mask"]
                codec_mask = batch["codec_mask"]
                speaker_positions = batch["speaker_positions"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(input_ids.device).to(model.dtype)
                ).detach()

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

                batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
                speaker_positions = speaker_positions.to(input_ids.device)
                input_codec_embedding[batch_indices, speaker_positions, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for index in range(1, model.talker.config.num_code_groups):
                    codec_index_embedding = model.talker.code_predictor.get_input_embeddings()[index - 1](
                        codec_ids[:, :, index]
                    )
                    codec_index_embedding = codec_index_embedding * all_codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_index_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]
                if talker_codec_ids.numel() == 0:
                    raise ValueError("Batch contains no target codec frames. Check the training JSONL.")

                _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids,
                    talker_hidden_states,
                )

                loss = outputs.loss + args.sub_talker_loss_weight * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_steps == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | "
                    f"CE: {outputs.loss.item():.4f} | Sub: {sub_talker_loss.item():.4f}"
                )

        _save_checkpoint(
            accelerator=accelerator,
            model=model,
            processor=qwen3tts.processor,
            output_model_path=args.output_model_path,
            epoch=epoch,
        )


if __name__ == "__main__":
    train()
