# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for TensorRT-LLM checkpoint export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List

from .tensorrt_llm_type import LayerNormPositionType, LayerNormType, MLPType

if TYPE_CHECKING:
    from transformers import T5Config

from modelopt import __version__

from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    QUANTIZATION_NONE,
    DecoderLayerConfig,
    MLPConfig,
    ModelConfig,
    MOEConfig,
)

# For NEMO and ENC/DEC models where the TensorRT-LLM model architecture and HF config are not aligned.
MODEL_NAME_TO_HF_ARCH_MAP = {
    "llama": "LlamaForCausalLM",
    "gpt": "GPTForCausalLM",
    "t5_encoder": "EncoderModel",
    "t5_decoder": "DecoderModel",
    "mllama": "MLLaMAModel",
}


def is_tensorrt_llm_0_8_or_9():
    """Returns true if tensorrt_llm version is 0.8 or 0.9."""
    try:
        import tensorrt_llm

        return tensorrt_llm.__version__.startswith(("0.8", "0.9"))
    except Exception:
        return False


def _find_layernorm_type(model_config: ModelConfig):
    if model_config.ln_f:
        return model_config.ln_f.layernorm_type
    for layer in model_config.layers:
        if layer.input_layernorm:
            return layer.input_layernorm.layernorm_type
        if layer.post_layernorm:
            return layer.post_layernorm.layernorm_type
    return LAYERNORM_DEFAULT


def _detect_exclude_modules(weight_keys: Iterable[str]) -> List[str]:
    quantized_layers = set()
    unquantized_layers = set()

    for key in weight_keys:
        suffix = key.split(".")[-1]

        if "_scaling_factor" in suffix:
            quantized_layers.add(key.rsplit(".", 1)[0])
        else:
            unquantized_layers.add(key.rsplit(".", 1)[0])

    return list(unquantized_layers - quantized_layers)


def _get_block_size(model_config: ModelConfig):
    """Return the first block size that is not zero if any."""
    for layer in model_config.layers:
        module_list = [layer.attention.qkv] if layer.attention else []
        if layer.mlp and isinstance(layer.mlp, MLPConfig):
            module_list.extend(
                [
                    layer.mlp.fc,
                    layer.mlp.proj,
                    layer.mlp.gate,
                ]
            )
        if layer.mlp and isinstance(layer.mlp, MOEConfig):
            module_list.extend([layer.mlp.experts.fc, layer.mlp.experts.proj])
        for m in module_list:
            if m is not None and m.awq_block_size != 0:
                return m.awq_block_size
    return 0


def convert_to_tensorrt_llm_config(
    model_config: ModelConfig,
    weight_keys: Iterable[str] = ["lm_head"],
):
    """Convert to TensorRT-LLM checkpoint config.

    Args:
        model_config: The model_config to convert.
        weight_keys: The iterable of string of weights exported to the tensorrt_llm checkpoint.
    """
    layernorm_type_map = {i.name: i.value for i in LayerNormType}
    layernorm_position_map = {i.name: i.value for i in LayerNormPositionType}
    mlp_type_map = {i.name: i.value for i in MLPType}

    decoder_type = model_config.layers[0].decoder_type
    tp_size = model_config.tensor_parallel
    pp_size = model_config.pipeline_parallel

    first_attention_config = None
    first_attention_decoder_config = None
    for decoder_layer in model_config.layers:
        first_attention_config = (
            decoder_layer.attention or decoder_layer.self_attention or decoder_layer.cross_attention
        )

        if first_attention_config is not None:
            first_attention_decoder_config = decoder_layer
            break

    assert (
        first_attention_config is not None and first_attention_decoder_config is not None
    ), "Model must have at least one attention block"

    config_architecture = model_config.architecture
    if not config_architecture:
        config_architecture = MODEL_NAME_TO_HF_ARCH_MAP[decoder_type]
    # For T5 model
    if decoder_type in ["t5"]:
        # For encoder
        if model_config.enc_dec == "enc":
            config_architecture = MODEL_NAME_TO_HF_ARCH_MAP["t5_encoder"]
        # For decoder
        else:
            config_architecture = MODEL_NAME_TO_HF_ARCH_MAP["t5_decoder"]
    config = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "architecture": config_architecture,
        "dtype": model_config.dtype,
        "logits_dtype": "float16" if model_config.dtype == "bfloat16" else model_config.dtype,
        "num_hidden_layers": len(model_config.layers) * pp_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_kv_heads,
        "hidden_size": model_config.hidden_size,
        "norm_epsilon": (
            first_attention_decoder_config.mlp_layernorm.eps
            if decoder_type in ["t5"]
            else first_attention_decoder_config.input_layernorm.eps
        ),
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.max_position_embeddings,
        "hidden_act": model_config.hidden_act,
        "use_parallel_embedding": True,
        "embedding_sharding_dim": 0,
        "quantization": {"quant_algo": None, "kv_cache_quant_algo": None},
        "mapping": {
            "world_size": tp_size * pp_size,
            "tp_size": tp_size,
            "pp_size": pp_size,
        },
        "head_size": first_attention_decoder_config.attention_head_size,
        "intermediate_size": first_attention_decoder_config.ffn_hidden_size_local * tp_size,
        "position_embedding_type": (
            "alibi" if first_attention_decoder_config.use_alibi else "rope_gpt_neox"
        ),
        "share_embedding_table": True if (model_config.lm_head is None and pp_size == 1) else False,
        "residual_mlp": first_attention_decoder_config.residual_mlp is not None,
        # Model Optimizer customized fields
        "bias": first_attention_config.dense.bias is not None,
        "rotary_pct": first_attention_decoder_config.rotary_pct,
        "rank": model_config.rank,
        "decoder": first_attention_decoder_config.decoder_type,
        "rmsnorm": _find_layernorm_type(model_config) == LAYERNORM_RMS,
        "lm_head_bias": model_config.lm_head is not None and model_config.lm_head.bias is not None,
    }

    if first_attention_decoder_config.rotary_base:
        config["rotary_base"] = first_attention_decoder_config.rotary_base

    if first_attention_decoder_config.rope_scaling:
        config["rotary_scaling"] = first_attention_decoder_config.rope_scaling

    if model_config.quantization == "fp8":
        config["quantization"].update({"quant_algo": "FP8"})
    elif model_config.quantization == "int4_awq":
        config["quantization"].update(
            {
                "quant_algo": "W4A16_AWQ",
                "group_size": _get_block_size(model_config),
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        )
    elif model_config.quantization == "w4a8_awq":
        config["quantization"].update(
            {
                "quant_algo": "W4A8_AWQ",
                "group_size": _get_block_size(model_config),
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        )
    elif model_config.quantization == "int8_sq":
        config["quantization"].update(
            {
                "quant_algo": "W8A8_SQ_PER_CHANNEL",
            }
        )
    elif model_config.quantization == "nvfp4":
        config["quantization"].update(
            {
                "quant_algo": "NVFP4",
                "group_size": _get_block_size(model_config),
            }
        )
    elif model_config.quantization == "nvfp4_awq":
        config["quantization"].update(
            {
                "quant_algo": "NVFP4_AWQ",
                "group_size": _get_block_size(model_config),
                "has_zero_point": False,
                "pre_quant_scale": True,
            }
        )
    elif model_config.quantization == QUANTIZATION_NONE:
        config["quantization"].update(
            {
                "quant_algo": None,
            }
        )
    else:
        config["quantization"].update(
            {
                "quant_algo": model_config.quantization,
            }
        )

    # Deprecate exclude modules for per layer export
    if model_config.quantization != QUANTIZATION_NONE:
        exclude_modules = _detect_exclude_modules(weight_keys)
        # In TRT LLM, the embedding table is shared for the following models, so lm_head quantization format
        # won't be automatically detected in the excluded_modules. We need to manually add it to the exclusions.
        if config["share_embedding_table"]:
            exclude_modules.append("lm_head")
        config["quantization"]["exclude_modules"] = exclude_modules

    if first_attention_config.kv_cache_dtype is not None:
        config["quantization"].update(
            {
                "kv_cache_quant_algo": first_attention_config.kv_cache_dtype,
            }
        )

    if decoder_type == "gpt2":
        config["position_embedding_type"] = "learned_absolute"
    elif decoder_type == "chatglm":
        config.update(
            {
                "position_embedding_type": "rope_gptj",
                "intermediate_size": model_config.layers[0].ffn_hidden_size_local * tp_size // 2,
                "max_position_embeddings": model_config.layers[0].seq_length,  # 32768
                "chatglm_version": model_config.layers[0].chatglm_version,
                "add_bias_linear": first_attention_config.dense.bias is not None,  # False
                "add_qkv_bias": first_attention_config.qkv.bias is not None,  # True
                "apply_query_key_layer_scaling": False,
                "apply_residual_connection_post_layernorm": model_config.layers[
                    0
                ].apply_residual_connection_post_layernorm,  # False
                "rope_ratio": model_config.layers[0].rope_ratio,
            }
        )
        # default rotary_pct of chatglm
        config["rotary_pct"] = 0.5
    elif decoder_type == "glm":
        config.update(
            {
                "chatglm_version": "glm",
                "add_bias_linear": model_config.layers[0].attention.dense.bias is not None,  # True
                "add_qkv_bias": model_config.layers[0].attention.qkv.bias is not None,  # True
                "apply_query_key_layer_scaling": False,
                "apply_residual_connection_post_layernorm": model_config.layers[
                    0
                ].apply_residual_connection_post_layernorm,  # False
                "position_embedding_type": "learned_absolute",
                "rope_ratio": model_config.layers[0].rope_ratio,
            }
        )
    elif decoder_type == "falcon":
        config.update(
            {
                "position_embedding_type": (
                    "alibi_with_scale" if model_config.layers[0].use_alibi else "rope_gpt_neox"
                ),
                "parallel_attention": model_config.layers[0].parallel_attention,
                "new_decoder_architecture": model_config.layers[0].new_decoder_architecture,
            }
        )
    elif decoder_type == "gptj":
        config.update(
            {
                "position_embedding_type": "rope_gptj",
                "rotary_dim": first_attention_config.rotary_dim,
            }
        )
    elif decoder_type == "mpt":
        config.update(
            {
                "clip_qkv": first_attention_config.clip_qkv,
                "alibi_bias_max": model_config.layers[0].alibi_bias_max,
            }
        )
    elif decoder_type == "qwen":
        intermediate_size = model_config.layers[0].ffn_hidden_size_local * tp_size
        qwen_type = "qwen"
        if model_config.layers[0].qwen_type:
            qwen_type = model_config.layers[0].qwen_type  # "qwen" or "qwen2"
        # Qwen version 1 has actual intermediate_size one half of what's in hf_config
        if qwen_type == "qwen":
            intermediate_size *= 2
        config.update(
            {
                "intermediate_size": intermediate_size,
                "seq_length": model_config.layers[0].seq_length,
                "qwen_type": (
                    model_config.layers[0].qwen_type if model_config.layers[0].qwen_type else "qwen"
                ),
            }
        )
    elif decoder_type == "phi":
        config["partial_rotary_factor"] = model_config.layers[0].partial_rotary_factor
    elif decoder_type == "gemma2":
        config["final_logit_softcapping"] = model_config.layers[0].final_logit_softcapping
        config["attn_logit_softcapping"] = model_config.layers[0].attn_logit_softcapping
        config["query_pre_attn_scalar"] = model_config.layers[0].query_pre_attn_scalar
        config["inter_layernorms"] = True
    elif decoder_type == "recurrentgemma":
        config["conv_kernel"] = 4
        config["state_size"] = 1
        config["state_dtype"] = "float32"
        config["rnn_hidden_size"] = model_config.layers[0].rnn_hidden_size
        config["rnn_conv_dim_size"] = model_config.layers[0].rnn_hidden_size
        config["logits_soft_cap"] = model_config.layers[0].logits_soft_cap
        config["emb_scale_by_sqrt_dim"] = model_config.layers[0].emb_scale_by_sqrt_dim
        config["layer_types"] = model_config.layers[0].layer_types
    elif decoder_type == "t5":
        config["position_embedding_type"] = "relative"
        config["share_embedding_table"] = getattr(model_config, "share_embedding_table")
        config["has_position_embedding"] = (
            False if not getattr(model_config, "position_embedding") else True
        )
        # fallback to RmsNorm if not specified
        layernorm_type = model_config.layers[0].mlp_layernorm.layernorm_type
        if not layernorm_type:
            layernorm_type = "RmsNorm"
        config["layernorm_type"] = layernorm_type_map[layernorm_type]
        config["has_attention_qkvo_bias"] = (
            False
            if not (
                model_config.layers[0].attention.qkv.bias
                if model_config.enc_dec == "enc"
                else model_config.layers[0].self_attention.qkv.bias
            )
            else True
        )
        config["has_mlp_bias"] = False if not model_config.layers[0].mlp.fc.bias else True
        config["has_model_final_layernorm"] = True if model_config.ln_f else False
        config["mlp_type"] = mlp_type_map[
            (
                "GatedMLP"
                if isinstance(model_config.layers[0].mlp, MLPConfig)
                and model_config.layers[0].mlp.gate
                else "MLP"
            )
        ]
        config["use_prompt_tuning"] = False
        config["has_position_embedding"] = False if not model_config.position_embedding else True
        config["has_embedding_layernorm"] = False if not model_config.ln_embed else True
        config["has_embedding_scale"] = False
        config["ffn_hidden_size"] = model_config.layers[0].mlp.fc.weight.shape[0]
        config["q_scaling"] = 1 / config["head_size"] ** 0.5
        config["layernorm_position"] = layernorm_position_map["pre_layernorm"]
        config["relative_attention"] = config["position_embedding_type"] == "relative"
        config["max_distance"] = model_config.layers[0].rel_attn_max_distance
        config["num_buckets"] = model_config.layers[0].rel_attn_num_buckets
        config["model_type"] = "t5"
        config["use_parallel_embedding"] = True
        config["use_implicit_relative_attention"] = False
        if model_config.enc_dec == "dec":
            config["rescale_before_lm_head"] = False
            config["encoder_hidden_size"] = model_config.encoder_hidden_size
            config["encoder_num_heads"] = model_config.encoder_num_heads
            config["encoder_head_size"] = model_config.encoder_head_size
            config["skip_cross_kv"] = False

    elif "phi3" in decoder_type:
        if not first_attention_decoder_config.moe_num_experts:
            config["intermediate_size"] = config["intermediate_size"] // 2  # fc and gate are merged
        config["original_max_position_embeddings"] = (
            first_attention_decoder_config.original_max_position_embeddings
        )
        if (
            model_config.layers[0].longrope_scaling_short_factors is not None
            and model_config.layers[0].longrope_scaling_long_factors is not None
        ):
            config["position_embedding_type"] = "long_rope"
            config["rotary_scaling"] = None
            config.update(
                {
                    "longrope_scaling_short_factors": model_config.layers[
                        0
                    ].longrope_scaling_short_factors,
                    "longrope_scaling_long_factors": model_config.layers[
                        0
                    ].longrope_scaling_long_factors,
                }
            )

        if (
            model_config.layers[0].longrope_short_mscale is not None
            and model_config.layers[0].longrope_long_mscale is not None
        ):
            config.update(
                {
                    "longrope_short_mscale": model_config.layers[0].longrope_short_mscale,
                    "longrope_long_mscale": model_config.layers[0].longrope_long_mscale,
                }
            )

    if decoder_type == "phi3small":
        config["mup_attn_multiplier"] = model_config.layers[0].mup_attn_multiplier
        config["mup_embedding_multiplier"] = model_config.layers[0].mup_embedding_multiplier
        config["mup_use_scaling"] = model_config.layers[0].mup_use_scaling
        config["mup_width_multiplier"] = model_config.layers[0].mup_width_multiplier
        config["blocksparse_block_size"] = model_config.layers[0].blocksparse_block_size
        config["blocksparse_homo_head_pattern"] = model_config.layers[
            0
        ].blocksparse_homo_head_pattern
        config["blocksparse_num_local_blocks"] = model_config.layers[0].blocksparse_num_local_blocks
        config["blocksparse_vertical_stride"] = model_config.layers[0].blocksparse_vertical_stride
        config["dense_attention_every_n_layers"] = model_config.layers[
            0
        ].dense_attention_every_n_layers
        config["gegelu_limit"] = model_config.layers[0].gegelu_limit

        # temp solution for phi3small, remove this after aligning the naming inside TRT-LLM
        config["num_kv_heads"] = model_config.num_kv_heads
        config["rotary_embedding_base"] = first_attention_decoder_config.rotary_base

    if decoder_type == "deci":
        config["block_configs"] = [layer.block_config for layer in model_config.layers]

    if decoder_type == "dbrx":
        config["clip_qkv"] = first_attention_decoder_config.clip_qkv

    if decoder_type == "mllama":
        num_layers = config["num_hidden_layers"]
        num_kv_heads = model_config.num_kv_heads
        cross_attention_layers = set(model_config.layers[0].cross_attention_layers)

        config["num_kv_heads_per_layer"] = [
            0 if i in cross_attention_layers else num_kv_heads for i in range(num_layers)
        ]
        config["num_kv_heads_per_cross_attn_layer"] = [
            num_kv_heads if i in cross_attention_layers else 0 for i in range(num_layers)
        ]

        config["cross_attention"] = True
        config["cross_attention_layers"] = model_config.layers[0].cross_attention_layers
        config["embed_vocab_size"] = model_config.vocab_size + 8
        vision_output_dim = model_config.layers[0].vision_output_dim
        config["vision_output_dim"] = vision_output_dim if vision_output_dim != 0 else 7680

    # For Mixtral and Arctic
    if first_attention_decoder_config.moe_num_experts:
        config["moe"] = {
            "num_experts": first_attention_decoder_config.moe_num_experts,
            "top_k": first_attention_decoder_config.moe_top_k,
            "normalization_mode": 1,  # ExpertScaleNormalizationMode.RENORMALIZE
        }

        if decoder_type == "phi3":
            config["moe"]["normalization_mode"] = 2  # ExpertScaleNormalizationMode.SPARSE_MIXER,
            config["moe"]["sparse_mixer_epsilon"] = (
                first_attention_decoder_config.sparse_mixer_epsilon
            )

        config["mapping"]["moe_tp_size"] = config["mapping"]["tp_size"]
        config["mapping"]["moe_ep_size"] = 1

    # Handle Medusa decoding
    # TODO (chenhany): when inference pp > 1; only last pp has medusa heads
    if model_config.medusa_heads is not None:
        config["base_architecture"] = config["architecture"]
        config["architecture"] = "MedusaForCausalLM"
        # NOTE: max_draft_len is related to the medusa tree len. Currently it is hardcoded to 63.
        config["max_draft_len"] = 63
        config["num_medusa_heads"] = len(model_config.medusa_heads)
        config["num_medusa_layers"] = len(model_config.medusa_heads[0].medusa_layers)

    return config


def prepare_enc_dec_export_dir(tensorrt_llm_config: Dict[str, Any], export_root: Path):
    """Prepare the export directory for encoder-decoder model."""
    # For encoder
    if tensorrt_llm_config["architecture"] == "EncoderModel":
        export_dir = export_root.joinpath("encoder")
    # For decoder
    else:
        export_dir = export_root.joinpath("decoder")
    return export_dir


def prepare_enc_dec_decoder_layer(
    layer_config: DecoderLayerConfig,
    model_config: "T5Config",
    enc_dec: str,
    layers: List[DecoderLayerConfig],
):
    """Prepare the config for each decoder layer of encoder-decoder model."""
    layer_config.rel_attn_max_distance = model_config.relative_attention_max_distance
    layer_config.rel_attn_num_buckets = model_config.relative_attention_num_buckets
    if enc_dec == "enc" and layer_config.attention.rel_attn_table is None:
        layer_config.attention.rel_attn_table = layers[0].attention.rel_attn_table
    elif enc_dec == "dec" and layer_config.self_attention.rel_attn_table is None:
        layer_config.self_attention.rel_attn_table = layers[0].self_attention.rel_attn_table
