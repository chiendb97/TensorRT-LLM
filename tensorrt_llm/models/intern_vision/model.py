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
                           index_select, select, shape, slice, unsqueeze, interpolate, identity, repeat_interleave)
from ...layers import MLP, BertAttention, Embedding, LayerNorm, Linear, Conv2d
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..modeling_utils import QuantConfig
from .config import InternVisionConfig
from .convert import (load_hf_intern_vision_base, load_weights_from_hf_model)


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
            shape=(1, 1, hidden_size)
        )

        self.patch_embedding = Conv2d(
            in_channels=3, out_channels=self.hidden_size, kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size)
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = Parameter(shape=(1, self.num_positions, self.hidden_size))

    def forward(self, pixel_values: Tensor):
        batch_size = pixel_values.size(0)
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Todo: Use expand tensor instead repeat or init class_embedding with max batch size
        # class_embeds = expand(self.class_embedding, expand_shape=[batch_size, 1, self.hidden_size])
        # class_embeds = self.class_embedding.value.repeat([4, 1, 1])
        class_embeds = repeat_interleave(self.class_embedding.value, repeats=4, dim=0)
        embeddings = concat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding.value
        return embeddings


class InternVisionEncoderLayer(Module):

    def __init__(self,
                 hidden_size,
                 intermediate_size,
                 num_attention_heads,
                 max_position_embeddings,
                 hidden_act='relu',
                 tp_group=None,
                 tp_size=1,
                 dtype=None):
        super().__init__()

        self.attention = BertAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            tp_group=tp_group,
            tp_size=tp_size,
            dtype=dtype)

        self.mlp = MLP(hidden_size=hidden_size,
                       ffn_hidden_size=intermediate_size,
                       hidden_act=hidden_act,
                       tp_group=tp_group,
                       tp_size=tp_size,
                       dtype=dtype)

        self.norm1 = LayerNorm(normalized_shape=hidden_size,
                               eps=1e-06,
                               dtype=dtype)

        self.norm2 = LayerNorm(normalized_shape=hidden_size,
                               eps=1e-06,
                               dtype=dtype)

        self.ls1 = Parameter(shape=[hidden_size])
        self.ls2 = Parameter(shape=[hidden_size])

    def forward(self,
                hidden_states,
                attention_mask=None,
                input_lengths=None,
                max_input_length=None):
        residual = hidden_states

        attention_output = self.attention(self.norm1(hidden_states) * self.ls1.value,
                                          attention_mask=attention_mask,
                                          input_lengths=input_lengths,
                                          max_input_length=max_input_length)

        hidden_states = residual + identity(attention_output)

        residual = hidden_states

        hidden_states = self.mlp(self.norm1(hidden_states) * self.ls2.value)

        hidden_states = residual + identity(hidden_states)

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

        assert cls.__name__ != "InternVisionBase", f"Never call from InternVisionBase class!"

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
        # bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
        bs_range = [4, 4, 4]
        pixel_values = Tensor(
            name='pixel_values',
            dtype=trt.float32,
            shape=[-1, 3, 448, 448],
            dim_range=OrderedDict([('batch_size', [bs_range])]),
        )

        inputs = {
            'pixel_values': pixel_values
        }

        return inputs


class InternVisionModel(InternVisionBase):

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)

        self.config = config
        self.max_position_embeddings = config.max_position_embeddings
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
                hidden_act=config.hidden_act,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                dtype=config.dtype) for _ in range(config.num_hidden_layers)
        ])

    def forward(self,
                pixel_values=None):
        hidden_states = self.embedding(pixel_values)
        self.register_network_output('embedding_output', hidden_states)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states)
            # keep the last layer output name as hidden_states
            if ((idx == (self.config.num_hidden_layers - 1)) and
                    (self.config.architecture in ["InternVisionModel"])):
                hidden_states.mark_output('hidden_states', self.config.dtype)
            else:
                self.register_network_output(f"layer_{idx}_output",
                                             hidden_states)

        return hidden_states


InternVisionModel = InternVisionModel
