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
from typing import Optional, Union
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
import transformers

from tensorrt_llm.models.modeling_utils import PretrainedModel
from ...parameter import Parameter

from ..._common import default_net
from ...functional import (ACT2FN, Tensor, concat, constant, cumsum, expand,
                           index_select, select, shape, slice, unsqueeze, interpolate, identity, repeat_interleave,
                           expand_dims, add, gelu)
from ...layers import MLP, BertAttention, Embedding, LayerNorm, Linear, Conv2d
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..modeling_utils import QuantConfig
from .config import InternVisionConfig
from .convert import (load_hf_intern_vision_base, load_weights_from_hf_model)
from ...logger import logger


class InternVisionEmbedding(Module):

    def __init__(self,
                 hidden_size,
                 image_size,
                 patch_size,
                 dtype=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size

        self.class_embedding = Parameter(
            shape=[1, 1, hidden_size]
        )

        self.patch_embedding = Conv2d(
            in_channels=3, out_channels=self.hidden_size, kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size)
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = Parameter(shape=(1, self.num_positions, self.hidden_size))

    def forward(self, pixel_values: Tensor):
        batch_size = shape(pixel_values, 0)
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.cast(target_dtype))  # shape = [*, channel, width, height]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        expand_shape = concat([batch_size,
                               shape(self.class_embedding.value, -2),
                               shape(self.class_embedding.value, -1)])

        class_embeds = expand(self.class_embedding.value, expand_shape)  # shape = [*, 1, grid, grid]
        embeddings = concat([class_embeds, patch_embeds], dim=1)  # shape = [*, width + 1, grid, grid]

        embeddings = embeddings + self.position_embedding.value
        return embeddings


class InternVisionEncoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 max_position_embeddings,
                 norm_epsilon,
                 hidden_act,
                 qkv_bias,
                 mapping: Mapping,
                 dtype=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.mapping = mapping
        self.dtype = dtype

        self.attn = BertAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            attention_head_size=self.hidden_size // num_attention_heads,
            num_kv_heads=num_attention_heads,
            bias=qkv_bias,
            tp_group=self.mapping.tp_group,
            tp_size=self.mapping.tp_size,
            cp_group=self.mapping.cp_group,
            cp_size=self.mapping.cp_size,
            dtype=self.dtype)

        self.mlp = MLP(hidden_size=self.hidden_size,
                       ffn_hidden_size=intermediate_size,
                       hidden_act=hidden_act,
                       tp_group=self.mapping.tp_group,
                       tp_size=self.mapping.tp_size,
                       dtype=self.dtype)

        self.norm1 = LayerNorm(normalized_shape=self.hidden_size,
                               eps=norm_epsilon,
                               dtype=self.dtype)

        self.norm2 = LayerNorm(normalized_shape=self.hidden_size,
                               eps=norm_epsilon,
                               dtype=self.dtype)

        self.ls1 = Parameter(shape=[self.hidden_size])

        self.ls2 = Parameter(shape=[self.hidden_size])

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states)) * self.ls1.value

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) * self.ls2.value

        return hidden_states


class InternVisionBase(PretrainedModel):
    '''
    Base class that provides from_huggingface() and prepare_inputs() methods
    '''
    config_class = InternVisionConfig

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

    @classmethod
    def load_hf_intern_vision(cls, model_dir: str, load_model_on_cpu: bool,
                              dtype: torch.dtype):
        """
        Use as the abstractmethod, load corresponding HF model.
        Subclass must implement this method!
        """

        if cls.__name__ == "InternVisionModel":
            return load_hf_intern_vision_base(model_dir, load_model_on_cpu, dtype)
        else:
            assert False, f"Unknown class {cls.__name__}!"

    @classmethod
    def from_hugging_face(
            cls,
            hf_model_or_dir: Union[str, 'transformers.PreTrainedModel'],
            dtype: str = 'float16',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        """
        Create a InternVision object from give parameters
        """
        import transformers

        assert hf_model_or_dir is not None
        use_preloading = isinstance(hf_model_or_dir,
                                    transformers.PreTrainedModel)
        if use_preloading:
            hf_model = hf_model_or_dir
            hf_config_or_dir = hf_model.config
        else:
            hf_model_dir = hf_model_or_dir
            hf_config_or_dir = hf_model_or_dir

        load_model_on_cpu = kwargs.pop('load_model_on_cpu', False)
        tllm_config = InternVisionConfig.from_hugging_face(
            hf_config_or_dir=hf_config_or_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs)

        setattr(tllm_config, 'architecture', cls.__name__)

        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        if not use_preloading:
            hf_model = cls.load_hf_intern_vision(model_dir=hf_model_dir,
                                                 load_model_on_cpu=load_model_on_cpu,
                                                 dtype=torch_dtype)
        weights = load_weights_from_hf_model(hf_model=hf_model,
                                             config=tllm_config)
        model = cls(tllm_config)
        model.load(weights)

        return model

    # Override the PretrainedModel's method, can unify in the future.
    def prepare_inputs(self, max_batch_size, **kwargs):
        # opt_shape is set to half of max batch_size and seq_len by default
        # tune this according to real data distribution
        bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        pixel_values = Tensor(
            name='pixel_values',
            dtype=trt.float32,
            shape=[-1, 3, 448, 448],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('in_channels', [[3, 3, 3]]),
                                   ('latent_height', [[448, 448, 448]]),
                                   ('latent_width', [[448, 448, 448]])]),
        )

        inputs = {
            'pixel_values': pixel_values
        }

        return inputs


class MLP1(Module):
    def __init__(self, hidden_size, downsample_ratio, llm_hidden_size):
        super().__init__()
        self.norm = LayerNorm(hidden_size * int(1 / downsample_ratio) ** 2)
        self.fc1 = Linear(hidden_size * int(1 / downsample_ratio) ** 2, llm_hidden_size)
        self.fc2 = Linear(llm_hidden_size, llm_hidden_size)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(self.norm(x))))


class InternVisionModel(InternVisionBase):

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

        self.config = config
        self.embedding = InternVisionEmbedding(
            hidden_size=config.hidden_size,
            image_size=config.image_size,
            patch_size=config.patch_size,
            dtype=config.dtype)

        self.layers = ModuleList([
            InternVisionEncoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_attention_heads=config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                norm_epsilon=config.norm_epsilon,
                hidden_act=config.hidden_act,
                qkv_bias=config.qkv_bias,
                mapping=config.mapping,
                dtype=config.dtype) for _ in range(config.num_hidden_layers)
        ])

        self.mlp1 = MLP1(config.hidden_size, config.downsample_ratio, config.llm_hidden_size)

    @staticmethod
    def pixel_shuffle(x: Tensor, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view([n, w, int(h * scale_factor), int(c / scale_factor)])
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute([0, 2, 1, 3])
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view([n, int(h * scale_factor), int(w * scale_factor),
                    int(c / (scale_factor * scale_factor))])

        x = x.permute([0, 2, 1, 3])
        return x

    def forward(self, pixel_values=None):
        hidden_states = self.embedding(pixel_values)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states)

        new_sizes = concat([shape(hidden_states, 0), shape(hidden_states, 1) - 1, shape(hidden_states, 2)])
        hidden_states = slice(hidden_states, starts=[0, 1, 0], sizes=new_sizes)

        h = w = int(hidden_states.size(1) ** 0.5)

        hidden_states = hidden_states.view([hidden_states.size(0), h, w, hidden_states.size(2)])

        hidden_states = self.pixel_shuffle(
            hidden_states, scale_factor=self.config.downsample_ratio)

        hidden_states = hidden_states.view(
            [hidden_states.size(0),
             int(h * self.config.downsample_ratio) * int(w * self.config.downsample_ratio),
             hidden_states.size(-1)])

        hidden_states = self.mlp1(hidden_states)

        hidden_states.mark_output('hidden_states', self.config.dtype)
        return hidden_states


InternVisionModel = InternVisionModel
