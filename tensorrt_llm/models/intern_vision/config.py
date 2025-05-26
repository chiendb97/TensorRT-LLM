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

import torch
import transformers

from ..._utils import torch_dtype_to_str
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class InternVisionConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 num_channels,
                 patch_size,
                 image_size,
                 downsample_ratio,
                 llm_hidden_size,
                 **kwargs):
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.downsample_ratio = downsample_ratio
        self.llm_hidden_size = llm_hidden_size

        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output['num_channels'] = self.num_channels
        output['patch_size'] = self.patch_size
        output['image_size'] = self.image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['llm_hidden_size'] = self.llm_hidden_size

        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        hf_config_dir = str(hf_config_or_dir)

        hf_config = transformers.AutoConfig.from_pretrained(hf_config_dir, trust_remote_code=True)

        num_key_value_heads = getattr(hf_config.vision_config, "num_key_value_heads",
                                      hf_config.vision_config.num_attention_heads)
        head_dim = getattr(
            hf_config.vision_config, "head_dim",
            hf_config.vision_config.hidden_size // hf_config.vision_config.num_attention_heads)
        head_size = getattr(hf_config.vision_config, "kv_channels", head_dim)

        # Todo: qk_normalization (only False)

        if dtype == 'auto':
            dtype = getattr(hf_config.vision_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'

        return cls(
            architecture=hf_config.vision_config.architectures[0],
            dtype=dtype,
            hidden_size=hf_config.vision_config.hidden_size,
            num_hidden_layers=hf_config.vision_config.num_hidden_layers,
            num_attention_heads=hf_config.vision_config.num_attention_heads,
            hidden_act=hf_config.vision_config.hidden_act,
            logits_dtype='float32',
            norm_epsilon=hf_config.vision_config.layer_norm_eps,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=hf_config.vision_config.intermediate_size,
            head_size=head_size,
            quantization=quant_config,
            mapping=mapping,
            #InternVision model args
            num_channels=hf_config.vision_config.num_channels,
            patch_size=hf_config.vision_config.patch_size,
            image_size=hf_config.vision_config.image_size,
            downsample_ratio=hf_config.downsample_ratio,
            llm_hidden_size=hf_config.llm_config.hidden_size,
            **kwargs)
