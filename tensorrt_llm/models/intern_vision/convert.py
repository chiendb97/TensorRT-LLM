# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
# isort: off
from transformers import AutoModel
# isort: on
from ...logger import logger
from ..convert_utils import split, split_qkv_bias_tp, split_qkv_tp
from .config import InternVisionConfig


def extract_layer_idx(name):
    ss = name.split('.')
    for s in ss:
        if s.isdigit():
            return s
    return None


def _load_weights_from_hf_intern_vision_model(hf_model,
                                              model_config: InternVisionConfig,
                                              torch_dtype: torch.dtype = torch.float16):
    weights = {}
    no_match = {}
    mapping = model_config.mapping
    # use different prefix because InternVision is used both individually and as part of model
    trtllm_prefix = ""
    for k, v in hf_model.state_dict().items():
        key = None
        v = v.to(torch_dtype).cpu()

        if 'mlp1.0.weight' in k:
            key = f'{trtllm_prefix}mlp1.norm.weight'
        elif 'mlp1.0.bias' in k:
            key = f'{trtllm_prefix}mlp1.norm.bias'
        elif 'mlp1.1.weight' in k:
            key = f'{trtllm_prefix}mlp1.fc1.weight'
        elif 'mlp1.1.bias' in k:
            key = f'{trtllm_prefix}mlp1.fc1.bias'
        elif 'mlp1.3.weight' in k:
            key = f'{trtllm_prefix}mlp1.fc2.weight'
        elif 'mlp1.3.bias' in k:
            key = f'{trtllm_prefix}mlp1.fc2.bias'
        elif 'vision_model.embeddings.class_embedding' in k:
            key = f'{trtllm_prefix}embedding.class_embedding'
        elif 'vision_model.embeddings.position_embedding' in k:
            key = f'{trtllm_prefix}embedding.position_embedding'
        elif 'vision_model.embeddings.patch_embedding.weight' in k:
            key = f'{trtllm_prefix}embedding.patch_embedding.weight'
        elif 'vision_model.embeddings.patch_embedding.bias' in k:
            key = f'{trtllm_prefix}embedding.patch_embedding.bias'
        elif "vision_model.encoder.layers" in k:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                no_match[k] = v
                continue
            idx = int(layer_idx)
            if 'ls1' in k:
                key = f'{trtllm_prefix}layers.{idx}.ls1'
            elif 'ls2' in k:
                key = f'{trtllm_prefix}layers.{idx}.ls2'
            elif 'norm1.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.norm1.weight'
            elif 'norm1.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.norm1.bias'
            elif 'norm2.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.norm2.weight'
            elif 'norm2.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.norm2.bias'
            elif 'mlp.fc1.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.fc.weight'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=0)
            elif 'mlp.fc1.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.fc.bias'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=0)
            elif 'mlp.fc2.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.proj.weight'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=1)
            elif 'mlp.fc2.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.mlp.proj.bias'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=0)
            elif 'attn.proj.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.attn.dense.weight'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=1)
            elif 'attn.proj.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.attn.dense.bias'
                v_clone = v.clone()
                v = split(v=v_clone,
                          tp_size=mapping.tp_size,
                          idx=mapping.tp_rank,
                          dim=0)
            elif 'attn.qkv.weight' in k:
                key = f'{trtllm_prefix}layers.{idx}.attn.qkv.weight'
                v_clone = v.clone()
                v = split_qkv_tp(v_clone,
                                 model_config.num_attention_heads,
                                 model_config.hidden_size,
                                 mapping.tp_size, mapping.tp_rank)
            elif 'attn.qkv.bias' in k:
                key = f'{trtllm_prefix}layers.{idx}.attn.qkv.bias'
                v_clone = v.clone()
                v = split_qkv_bias_tp(v_clone,
                                      model_config.num_attention_heads,
                                      model_config.hidden_size,
                                      mapping.tp_size, mapping.tp_rank)
            else:
                no_match[k] = v
                continue
        else:
            no_match[k] = v
            continue

        weights[key] = v

    return (weights, no_match)


def load_hf_intern_vision_base(model_dir: str,
                               load_model_on_cpu: bool = False,
                               dtype: torch.dtype = torch.float16):


    """
    load huggingface BertModel and RobertaModel model
    """

    # Todo: Refactor
    def _get_pos_embed(pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, 448 // 14, 448 // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def get_position_embedding(position_embedding):
        position_embedding = torch.cat([
            position_embedding[:, :1, :],
            _get_pos_embed(position_embedding[:, 1:, :], 32, 32)
        ], dim=1)

        return nn.Parameter(position_embedding)

    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    # Prepare position embedding with fixed image size
    model.vision_model.embeddings.position_embedding = get_position_embedding(model.vision_model.embeddings.position_embedding)

    if not load_model_on_cpu:
        model.cuda().to(dtype)
    model.eval()
    return model


def load_weights_from_hf_model(
        hf_model,
        config: InternVisionConfig,
):
    """
    load trtllm weights from hf model

    return a dict of weights, with trtllm weights naming

    """
    # TODO: add quantization support
    weights = {}
    tik = time.time()

    torch_dtype = getattr(torch, config.dtype)

    no_match = {}
    if config.architecture in [
        "InternVLChatModel", "InternVisionModel"
    ]:
        weights, no_match = _load_weights_from_hf_intern_vision_model(
            hf_model=hf_model, model_config=config, torch_dtype=torch_dtype)
    else:
        assert False, f"Unknown Intern Vision model {config.architecture}"

    if no_match:
        logger.warning(
            f"These weights from huggingface model are not used:\n {[key for key in no_match.keys()]}"
        )

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: InternVisionConfig,
             device: str = 'cuda',
             calib_dataset: str = 'cnn_dailymail'):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    logger.warning(f"FP8 Support for Intern Vision will come soon!")
