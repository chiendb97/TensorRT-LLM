# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Code that export quantized Hugging Face models for deployment."""

import json
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from modelopt import __version__
from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

from .layer_utils import is_moe, is_quantlinear, update_experts_avg_prequant_scale
from .model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_INT4_AWQ,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
)
from .quantization_utils import (
    _requantize_fused_modules,
    _resmooth_fused_modules,
    _resmooth_module_and_update_params,
    convert_state_dict_amax_to_scales,
    filter_output_quantizer,
    get_activation_scaling_factor,
    get_kv_cache_dtype,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    process_layer_quant_config,
    to_quantized_weight,
)

__all__ = ["export_hf_checkpoint"]

SPECULATIVE_DECODING_MODULE_NAMES = ["medusa_heads", "eagle_module", "drafter"]


def requantize_resmooth_fused_llm_layers(model: torch.nn.Module):
    """Group modules that take the same input and register shared parameters in module."""
    # TODO: Handle DBRX MoE
    input_to_linear = defaultdict(list)
    quantization_format = get_quantization_format(model)

    def _hook(module, input, output):
        """Update dictionary with list of all modules that share the same input."""
        # TODO: Handle DBRX MoE case
        input_to_linear[input].append(module)

    handles = []

    for _, module in model.named_modules():
        # For MoE models update pre_quant_scale to average pre_quant_scale amongst experts
        if is_moe(module) and (
            quantization_format in [QUANTIZATION_NVFP4_AWQ, QUANTIZATION_INT4_AWQ]
        ):
            update_experts_avg_prequant_scale(module)

        elif "QuantLinear" in type(module).__name__:
            handle = module.register_forward_hook(_hook)
            handles.append(handle)

    with torch.no_grad():
        fake_input = torch.ones([1, 1], dtype=torch.long).to(model.device)
        # Run forward pass so that all modules sharing the same input are collected using forward hook.
        model(fake_input)
        for handle in handles:
            handle.remove()

    for _, modules in input_to_linear.items():
        # Compute and register shared parameters only for fused layers
        if len(modules) > 1:
            _resmooth_fused_modules(modules)
            _requantize_fused_modules(modules)


def _export_hf_checkpoint(
    model: nn.Module, dtype: Optional[torch.dtype] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.

    Args:
        model: the torch model.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.

    Returns:
        post_state_dict: Dict containing quantized weights
        quant_config: config information to export hf_quant_cfg.json
        per_layer_quantization: Dict containing layerwise quantization information to export quant_cfg.json
        in mixed_precision case.
    """
    if dtype is None:
        dtype = model.config.torch_dtype
    else:
        warnings.warn(
            f"Model's original dtype ({model.config.torch_dtype}) differs from target dtype "
            f"({dtype}), which may lead to numerical errors."
        )

    # Base model layers
    layer_pool = {
        f"model.layers.{name}": sub_module
        for name, sub_module in model.model.layers.named_modules()
    }
    # NOTE: Speculative decoding models have extra modules that may be quantized
    # Need to add these modules to the layer_pool
    for key in SPECULATIVE_DECODING_MODULE_NAMES:
        if hasattr(model, key):
            for name, sub_module in getattr(model, key).named_modules():
                layer_pool.update({f"{key}.{name}": sub_module})

    # Layer config dict holds quantization format of each layer.
    # It also holds awq_block_size information for applicable layers.
    layer_config_dict = {}

    # Resmooth and requantize fused layers
    # TODO: Handle mixed precision
    requantize_resmooth_fused_llm_layers(model)

    for name, sub_module in layer_pool.items():
        if is_quantlinear(sub_module):
            quantization_format = get_quantization_format(sub_module)
            block_size = get_weight_block_size(sub_module)

            # Construct per layer config dictionary
            layer_config_dict.update({name + ".quantization": quantization_format})
            layer_config_dict.update({name + ".awq_block_size": block_size})

            if quantization_format == QUANTIZATION_FP8:
                maxbound = sub_module.weight_quantizer.maxbound

                # Convert amax to float32
                sub_module.weight_quantizer._amax = sub_module.weight_quantizer._amax.to(
                    torch.float32
                )

                if sub_module.weight_quantizer._amax.dim() == 1:
                    weight_scaling_factor = torch.tensor(
                        sub_module.weight_quantizer.amax.item()
                        / sub_module.weight_quantizer.maxbound
                    )
                else:
                    # Per-channel amax
                    weight_scaling_factor = torch.tensor(
                        sub_module.weight_quantizer.amax / sub_module.weight_quantizer.maxbound
                    )

                sub_module.register_buffer(
                    "weight_scale",
                    weight_scaling_factor,
                )

                if hasattr(sub_module.input_quantizer, "_amax"):
                    sub_module.input_quantizer._amax = sub_module.input_quantizer._amax.to(
                        torch.float32
                    )

                    sub_module.register_buffer(
                        "input_scale",
                        get_activation_scaling_factor(sub_module),
                    )

                if hasattr(sub_module.output_quantizer, "_amax"):
                    sub_module.output_quantizer._amax = sub_module.output_quantizer._amax.to(
                        torch.float32
                    )

            elif quantization_format in [QUANTIZATION_NVFP4_AWQ, QUANTIZATION_NVFP4]:
                # We do not scale amax values during postprocessing for NVFP4_AWQ or NVFP4.
                maxbound = 1.0

                # Check if weight_scale_2 is precomputed. Shared weight_scale_2 is already registered for fused layers.
                if not hasattr(sub_module, "weight_scale_2"):
                    # Resmooth weights for unfused experts
                    if hasattr(sub_module.input_quantizer, "experts_avg_pre_quant_scale"):
                        _resmooth_module_and_update_params(
                            sub_module, sub_module.input_quantizer.experts_avg_pre_quant_scale
                        )

                    # Register weight_scale_2
                    sub_module.register_buffer(
                        "weight_scale_2",
                        get_weight_scaling_factor_2(sub_module).reshape(1),
                    )
                    # Register quantized weight_scale
                    sub_module.register_buffer(
                        "weight_scale",
                        NVFP4QTensor.get_weights_scaling_factor(
                            sub_module.weight, block_size, sub_module.weight_scale_2
                        ),
                    )

                # Record activation_scaling_factor
                sub_module.register_buffer(
                    "input_scale",
                    get_activation_scaling_factor(sub_module),
                )

            elif quantization_format == QUANTIZATION_INT4_AWQ:
                # INT4_AWQ scales quantizer amax values using maxbound during postprocessing
                maxbound = sub_module.weight_quantizer.maxbound
                # Resmooth weights for unfused experts
                if hasattr(sub_module.input_quantizer, "experts_avg_pre_quant_scale"):
                    _resmooth_module_and_update_params(
                        sub_module, sub_module.input_quantizer.experts_avg_pre_quant_scale
                    )

                sub_module.register_buffer("weight_scale", get_weight_scaling_factor(sub_module))
                sub_module.register_buffer("input_scale", get_activation_scaling_factor(sub_module))

            # Check if quantization format is None, to support auto_quant
            if quantization_format != QUANTIZATION_NONE:
                quantized_weight = to_quantized_weight(
                    sub_module.weight.to(dtype),
                    sub_module.weight_scale,
                    quantization_format,  # type:ignore [arg-type]
                    sub_module.weight_scale_2 if hasattr(sub_module, "weight_scale_2") else None,
                    block_size,
                )
                sub_module.weight = nn.Parameter(quantized_weight, requires_grad=False)

            # Find kv cache quant format
            kv_cache_format = get_kv_cache_dtype(sub_module)

    quantized_state_dict = model.state_dict()

    # If kv cache quantization enabled, only keep output quantizers for kv_cache
    if kv_cache_format is not QUANTIZATION_NONE:
        quantized_state_dict = filter_output_quantizer(quantized_state_dict)

    # Process per layer quantization config dict
    per_layer_quantization = process_layer_quant_config(layer_config_dict)

    # Convert the amax to scales
    if not per_layer_quantization:
        layers_quant_config = quantization_format
    else:
        layers_quant_config = layer_config_dict

    # TODO: Handle maxbound separately for kv cache quantization
    post_state_dict = convert_state_dict_amax_to_scales(
        quantized_state_dict, maxbound, layers_quant_config
    )

    # Create the quantization config
    # TODO: add support for customized mixed precision config
    quant_config: Dict[str, Any] = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "quantization": {"quant_algo": None, "kv_cache_quant_algo": None},
    }

    if quantization_format == "fp8":
        quant_config["quantization"].update({"quant_algo": "FP8", "exclude_modules": ["lm_head"]})
        # TODO: add info about per-channel weight scale and dynamic per token input scale
    elif quantization_format == "int4_awq":
        quant_config["quantization"].update(
            {
                "quant_algo": "W4A16_AWQ",
                "group_size": block_size,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["lm_head"],
            }
        )
    elif quantization_format == "nvfp4":
        quant_config["quantization"].update(
            {
                "quant_algo": "NVFP4",
                "group_size": block_size,
                "exclude_modules": ["lm_head"],
            }
        )
    elif quantization_format == "nvfp4_awq":
        quant_config["quantization"].update(
            {
                "quant_algo": "NVFP4_AWQ",
                "group_size": block_size,
                "has_zero_point": False,
                "pre_quant_scale": True,
                "exclude_modules": ["lm_head"],
            }
        )
    else:
        quant_config["quantization"].update(
            {
                "quant_algo": (
                    quantization_format if quantization_format != QUANTIZATION_NONE else None
                ),
            }
        )

    if kv_cache_format is not None:
        quant_config["quantization"].update(
            {
                "kv_cache_quant_algo": kv_cache_format,
            }
        )

    return post_state_dict, quant_config, per_layer_quantization


def export_hf_checkpoint(
    model: nn.Module,
    dtype: Optional[torch.dtype] = None,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
):
    """Exports the torch model to unified checkpoint and saves to export_dir.

    Args:
        model: the torch model.
        dtype: the weights data type to export the unquantized layers or the default model data type if None.
        export_dir: the target export path.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    try:
        post_state_dict, hf_quant_config, per_layer_quantization = _export_hf_checkpoint(
            model, dtype
        )

        # If auto_quant is used, save per layer quantization information in quant_cfg.json
        if per_layer_quantization:
            # Update auto quant related information in quantization.json
            per_layer_quantization["kv_cache_quant_algo"] = hf_quant_config["quantization"][
                "kv_cache_quant_algo"
            ]

            # Update auto quant related informaion in hf_quant_config.json
            # We remove group_size, has_zero_point and pre_quant_scale information from config.json
            hf_quant_config["quantization"] = {
                k: per_layer_quantization[k] for k in ("quant_algo", "kv_cache_quant_algo")
            }

            with open(f"{export_dir}/quant_cfg.json", "w") as file:
                json.dump(per_layer_quantization, file, indent=4)

        # Save config
        with open(f"{export_dir}/hf_quant_config.json", "w") as file:
            json.dump(hf_quant_config, file, indent=4)

        # Save model
        model.save_pretrained(export_dir, state_dict=post_state_dict)

    except Exception as e:
        fallback_model_path = f"{export_dir}/modelopt_model.pth"
        torch.save(model.state_dict(), fallback_model_path)
        warnings.warn(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            f" (including the quantization factors) is saved to {fallback_model_path} using"
            " torch.save for further inspection."
        )
        raise e
