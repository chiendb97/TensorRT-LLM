# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Code from TRT-LLM that export optimized models to the TensorRT-LLM checkpoint."""

from enum import IntEnum


# These clasess are directly copied from TRT-LLM to relex TRT-LLM dependency for checkpoint export
class LayerNormType(IntEnum):
    """LayerNormType from tensorrt_llm.functional."""

    LayerNorm = 0
    RmsNorm = 1
    GroupNorm = 2


class LayerNormPositionType(IntEnum):
    """LayerNormPositionType from tensorrt_llm.functional."""

    pre_layernorm = 0
    post_layernorm = 1


class MLPType(IntEnum):
    """MLPType from tensorrt_llm.functional."""

    MLP = 0
    GatedMLP = 1
    FusedGatedMLP = 2
