# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Utils for model_config export.

Some of the logics in this file are empirical and needs constant update if exceptions occur.
"""

from typing import List, Optional, Tuple
from warnings import warn

import torch
import torch.nn as nn

try:
    from transformers.activations import ACT2FN
except Exception:
    warn("Cannot find transformers package. Hugginface modules cannot be exported.")
    pass

from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
from modelopt.torch.utils import distributed as dist

from .hf_config_map import HF_CONFIG_MAP
from .mcore_config_map import MCORE_CONFIG_MAP
from .model_config import (
    LAYERNORM_DEFAULT,
    LAYERNORM_RMS,
    LINEAR_COLUMN,
    LINEAR_GROUP,
    LINEAR_ROW,
    QUANTIZATION_FP8,
    QUANTIZATION_NONE,
    QUANTIZATION_NVFP4,
    QUANTIZATION_NVFP4_AWQ,
    AttentionConfig,
    ConvConfig,
    DecoderLayerConfig,
    EmbeddingConfig,
    ExpertConfig,
    LayernormConfig,
    LinearActConfig,
    LinearConfig,
    MedusaHeadConfig,
    MLPConfig,
    MOEConfig,
    QKVConfig,
    RecurrentConfig,
    RelativeAttentionTableConfig,
    RgLruConfig,
)
from .model_config_utils import pad_weights
from .quantization_utils import (
    get_activation_scaling_factor,
    get_kv_cache_dtype,
    get_kv_cache_scaling_factor,
    get_prequant_scaling_factor,
    get_quantization_format,
    get_weight_block_size,
    get_weight_scaling_factor,
    get_weight_scaling_factor_2,
    resmooth_and_get_scale_and_amax,
)

try:
    from megatron.core.transformer.module import MegatronModule

    has_mcore = True
except ImportError:
    has_mcore = False


def get_dtype(model):
    """Returns the default dtype of the model."""
    for weight in model.parameters():
        if torch.is_floating_point(weight):
            return weight.dtype
    return None


def check_model_compatibility(module_list: List[nn.Module]) -> Tuple[bool, bool, bool]:
    """Returns whether the list of modules is compatible with the export logic.

    And if positional embedding and embedding layernorm exists.

    We assumes the model to be assembled with one or two embedding layers,
    a ModuleList of transformer decoders,
    and a final layernorm with optional embedding layernorm.
    Otherwise it will not be supported.
    """
    num_embeddings = 0
    num_module_list = 0
    num_layer_norm = 0
    for module in module_list:
        if is_embedding(module):
            num_embeddings += 1
        elif is_decoder_list(module):
            num_module_list += 1
        elif is_layernorm(module):
            num_layer_norm += 1

    return (
        1 <= num_embeddings
        and num_embeddings <= 2
        and num_module_list == 1
        and 1 <= num_layer_norm
        and num_layer_norm <= 2,
        num_embeddings > 1,
        num_layer_norm > 1,
    )


def get_transformer_layers(model: nn.Module) -> List[nn.Module]:
    """Returns the root module of the transformer model."""
    if "Megatron" in type(model).__name__:
        if hasattr(model, "model") and "GPTModel" in type(model.model).__name__:
            # NEMO mcore models can be handled with the following branch.
            model = model.model

        # NEMO non mcore models, we need to find the language_model module first.
        children = [model]
        language_model = None
        while children and not language_model:
            next_children = []
            for child in children:
                if type(child).__name__ == "TransformerLanguageModel":
                    language_model = child
                    break
                for m in child.children():
                    next_children.append(m)
            children = next_children
        if language_model:
            warn("Warning: this is an old NEMO checkpoint format and will be deprecated soon.")
            layers = [m for m in language_model.embedding.children()] + [
                m for m in language_model.encoder.children()
            ]

            if hasattr(language_model, "output_layer"):
                layers.append(language_model.output_layer)

            return layers

    if "GPTModel" in type(model).__name__:
        # mcore models
        layers = []
        if hasattr(model, "embedding"):
            layers = layers + [m for m in model.embedding.children()]
        layers = layers + [m for m in model.decoder.children()]
        if hasattr(model, "output_layer"):
            layers.append(model.output_layer)
        return layers

    if hasattr(model, "glm"):
        model = model.glm

    if hasattr(model, "transformer"):
        # This is a LMHead model
        # Add lm_head to be processed along with transformer layers
        modules = []
        for m in model.transformer.children():
            # QwenVL's visual encoder name as 'VisionTransformer' has no `layers`.
            if (
                "Transformer" in type(m).__name__
                and hasattr(m, "layers")
                and is_decoder_list(m.layers)
            ):
                modules.append(m.layers)
                modules.append(m.final_layernorm)
            else:
                modules.append(m)
        if hasattr(model, "lm_head"):
            modules += [model.lm_head]
        return modules

    if hasattr(model, "model"):
        # LLAMA, InternLM2
        modules = [m for m in model.model.children()]
        # LLAMA
        if hasattr(model, "lm_head"):
            modules += [model.lm_head]
        # InternLM2
        elif hasattr(model, "output"):
            modules += [model.output]

        return modules

    return [m for m in model.children()]


def is_linear(module: nn.Module) -> bool:
    """Returns whether the module is a linear layer."""
    return any([k in type(module).__name__ for k in ["Linear", "Conv1D", "NormHead"]])


def is_embedding(module: nn.Module) -> bool:
    """Returns whether the module is an embedding layer."""
    module_type_name = type(module).__name__
    return (
        "Embedding" in module_type_name
        and "Rotary" not in module_type_name
        and "PhiImage" not in module_type_name
        and "Phi3Image" not in module_type_name
    )


def build_embedding_config(
    module: nn.Module, dtype: torch.dtype, normalization_constant: float = 1
) -> EmbeddingConfig:
    """Builds the embedding config from the module."""
    assert is_embedding(module)

    world_size = dist.size()
    rank = dist.rank()

    # Special case for chatglm
    if hasattr(module, "word_embeddings"):
        module = module.word_embeddings

    weight = module.weight.type(dtype)
    normalized_weight = weight * normalization_constant
    if "Parallel" in type(module).__name__:
        local_weight = normalized_weight
    else:
        padded_weight = pad_weights(normalized_weight, dist.size())
        local_weight = torch.chunk(padded_weight, world_size, dim=0)[rank]
    return EmbeddingConfig(weight=local_weight, quantization=get_quantization_format(module))


def is_layernorm(module: nn.Module) -> bool:
    """Returns whether the module is a layernorm layer."""
    module_name = type(module).__name__
    return any(norm in module_name for norm in ["LayerNorm", "RMSNorm"])


def build_layernorm_config(module: nn.Module, dtype: torch.dtype) -> LayernormConfig:
    """Builds the layernorm config from the module."""
    assert is_layernorm(module)

    layernorm_type = LAYERNORM_DEFAULT
    if "RMS" in type(module).__name__:
        layernorm_type = LAYERNORM_RMS

    weight = module.weight

    def _weights_plus_one(module):
        if any(
            name in type(module).__name__
            for name in ["LayerNorm1P", "GemmaRMSNorm", "Gemma2RMSNorm"]
        ):
            return True

        if hasattr(module, "zero_centered_gamma") and module.zero_centered_gamma:
            return True

        return False

    if _weights_plus_one(module):
        # megatron layernorm's weight needs to be updated.
        weight = weight.float() + 1.0

    config = LayernormConfig(
        weight=weight.type(dtype),
        bias=(
            module.bias.type(dtype) if hasattr(module, "bias") and module.bias is not None else None
        ),
        layernorm_type=layernorm_type,
        quantization=get_quantization_format(module),
    )

    # TODO: handle the nemo llama eps config.
    for eps_key in ["eps", "variance_epsilon"]:
        if hasattr(module, eps_key):
            config.eps = getattr(module, eps_key)
            break

    return config


def is_decoder_list(module: nn.Module) -> bool:
    """Returns whether the module is a decoder list."""
    return type(module) is nn.ModuleList


def is_attention(module: nn.Module) -> bool:
    """Returns whether the module is an attention layer."""
    return "Attention" in type(module).__name__


def is_mlp(module: nn.Module) -> bool:
    """Returns whether the module is an MLP layer."""
    return any([key in type(module).__name__.upper() for key in ("MLP", "T5DENSE")])


def is_moe(module: nn.Module) -> bool:
    """Returns whether the module is an MOE layer."""
    return type(module).__name__ in [
        "MixtralSparseMoeBlock",
        "ArcticMoE",
        "DbrxFFN",
        "MoELayer",
        "PhiMoESparseMoeBlock",
    ]


def is_quantlinear(module: nn.Module) -> bool:
    """Returns whether the module is a quantized linear layer."""
    return "QuantLinear" in type(module).__name__


def dup_kv_weight(v: torch.Tensor, head_size: int, num_head: int, tp_size: int) -> torch.Tensor:
    """Repeat kv heads if tp_size > num_kv_heads."""
    assert tp_size % num_head == 0
    reps = tp_size // num_head
    v = v.view(-1, head_size, v.size(-1))
    v = v.repeat_interleave(reps, dim=0)
    v = v.view(-1, v.size(-1))
    return v.contiguous()


def build_qkv(
    qkv_modules: List[nn.Module],
    model_metadata_config,
    dtype: torch.dtype,
    ext_config: DecoderLayerConfig = None,
    tp_size: int = 1,
) -> QKVConfig:
    """Converts the qkv modules to the config."""
    config = QKVConfig()
    q_bias = None
    k_bias = None
    v_bias = None

    block_size = get_weight_block_size(qkv_modules[0])
    quantization = get_quantization_format(qkv_modules[0])

    num_heads = ext_config.num_attention_heads
    training_tp = model_metadata_config["training_tensor_parallel"]
    if len(qkv_modules) == 1:
        # QKV layers combined as a single module, e.g. gpt
        qkv_module = qkv_modules[0]
        assert ext_config is not None, "ext_config is None"
        num_kv_heads = ext_config.num_kv_heads

        if "ColumnParallelLinear" in type(qkv_module).__name__:
            # For NEMO model, num_kv_heads/num_attention_heads is the first dimension of QKV
            model_metadata_config["head_is_first_dim"] = True

        qkv_weight = qkv_module.weight
        if type(qkv_module).__name__ == "Conv1D":
            if not hasattr(qkv_module, "input_quantizer") and not hasattr(
                qkv_module, "output_quantizer"
            ):
                # For unquantized nn.Conv1D, the weights are transposed compared with the nn.Linear
                qkv_weight = qkv_weight.T

        # Handle the case that num_kv_heads/num_attention_heads is the first dimension of QKV.
        # This logic covers MQA and GQA as well.
        keep_channel_order = not model_metadata_config.get("head_is_first_dim", False)

        hidden_size = qkv_module.weight.shape[1]
        q_weight, k_weight, v_weight = _split_fused_qkv_weight_and_scaling(
            qkv_weight,
            hidden_size,
            num_heads,
            num_kv_heads,
            training_tp,
            False,
            keep_channel_order,
        )
        qkv_activation_scaling_factor = get_activation_scaling_factor(qkv_module)
        q_activation_scaling_factor = qkv_activation_scaling_factor
        k_activation_scaling_factor = qkv_activation_scaling_factor
        v_activation_scaling_factor = qkv_activation_scaling_factor

        qkv_weight_scaling_factor = get_weight_scaling_factor(qkv_module)

        if qkv_weight_scaling_factor is not None and qkv_weight_scaling_factor.numel() != 1:
            # INT8 sq case
            (
                q_weight_scaling_factor,
                k_weight_scaling_factor,
                v_weight_scaling_factor,
            ) = _split_fused_qkv_weight_and_scaling(
                qkv_weight_scaling_factor,
                hidden_size,
                num_heads,
                num_kv_heads,
                training_tp,
                True,
                keep_channel_order,
            )
        else:
            q_weight_scaling_factor = qkv_weight_scaling_factor
            k_weight_scaling_factor = qkv_weight_scaling_factor
            v_weight_scaling_factor = qkv_weight_scaling_factor

        # bias
        if qkv_module.bias is not None:
            q_bias, k_bias, v_bias = _split_fused_qkv_weight_and_scaling(
                qkv_module.bias,
                hidden_size,
                num_heads,
                num_kv_heads,
                training_tp,
                True,
                keep_channel_order,
            )

        q_weight_scaling_factor_2 = k_weight_scaling_factor_2 = v_weight_scaling_factor_2 = (
            get_weight_scaling_factor_2(qkv_module)
        )

        q_prequant_scaling_factor = k_prequant_scaling_factor = v_prequant_scaling_factor = (
            get_prequant_scaling_factor(qkv_module, dtype)
        )

        q_quantization = k_quantization = v_quantization = get_quantization_format(qkv_module)

    elif len(qkv_modules) == 3:
        # Separate QKV layers
        q_weight = qkv_modules[0].weight
        q_activation_scaling_factor = get_activation_scaling_factor(qkv_modules[0])
        q_weight_scaling_factor = get_weight_scaling_factor(qkv_modules[0])
        q_quantization = get_quantization_format(qkv_modules[0])
        k_weight = qkv_modules[1].weight
        k_activation_scaling_factor = get_activation_scaling_factor(qkv_modules[1])
        k_weight_scaling_factor = get_weight_scaling_factor(qkv_modules[1])
        k_quantization = get_quantization_format(qkv_modules[1])
        v_weight = qkv_modules[2].weight
        v_activation_scaling_factor = get_activation_scaling_factor(qkv_modules[2])
        v_weight_scaling_factor = get_weight_scaling_factor(qkv_modules[2])
        v_quantization = get_quantization_format(qkv_modules[2])

        q_weight_scaling_factor_2 = get_weight_scaling_factor_2(qkv_modules[0])
        k_weight_scaling_factor_2 = get_weight_scaling_factor_2(qkv_modules[1])
        v_weight_scaling_factor_2 = get_weight_scaling_factor_2(qkv_modules[2])

        q_prequant_scaling_factor = get_prequant_scaling_factor(qkv_modules[0], dtype)
        k_prequant_scaling_factor = get_prequant_scaling_factor(qkv_modules[1], dtype)
        v_prequant_scaling_factor = get_prequant_scaling_factor(qkv_modules[2], dtype)

        if hasattr(qkv_modules[0], "bias"):
            q_bias = qkv_modules[0].bias

        if hasattr(qkv_modules[1], "bias"):
            k_bias = qkv_modules[1].bias

        if hasattr(qkv_modules[2], "bias"):
            v_bias = qkv_modules[2].bias

    else:
        raise NotImplementedError(f"QKV modules format {qkv_modules} not supported")

    # derive num_kv_heads
    head_size = q_weight.size(0) // num_heads
    num_kv_heads = ext_config.num_kv_heads or k_weight.size(0) // head_size
    if tp_size > num_kv_heads:
        if any(
            map(
                lambda t: t is not None and t.numel() > 1,
                [
                    k_activation_scaling_factor,
                    k_weight_scaling_factor,
                    k_weight_scaling_factor_2,
                    k_bias,
                    v_activation_scaling_factor,
                    v_weight_scaling_factor,
                    v_weight_scaling_factor_2,
                    v_bias,
                ],
            )
        ):
            # TODO(oargov): handle cases with biases / scales
            raise NotImplementedError(
                "Duplicating KV heads for KV with non-scalar scales and/or biases is not supported"
            )

        # duplicte kv heads as needed
        k_weight = dup_kv_weight(k_weight, head_size, num_kv_heads, tp_size)
        v_weight = dup_kv_weight(v_weight, head_size, num_kv_heads, tp_size)

    # Adopt the implementation from examples/llama/weight.py in the tekit repo for INT4 AWQ
    # Resmooth q, k, v for int4_awq, as they share the same pre_quant_scale during compulation
    # This logic is implemented at the export stage to reduce resource requirement for model building/deployment
    if all(
        pre_quant_scale is not None
        for pre_quant_scale in [
            q_prequant_scaling_factor,
            k_prequant_scaling_factor,
            v_prequant_scaling_factor,
        ]
    ):
        pre_quant_scale = (
            q_prequant_scaling_factor + k_prequant_scaling_factor + v_prequant_scaling_factor
        ) / 3.0
        # Resmooth q, k, v with average pre_quant_scale for AWQ
        weights = [q_weight, k_weight, v_weight]
        prequant_scaling_factors = [
            q_prequant_scaling_factor,
            k_prequant_scaling_factor,
            v_prequant_scaling_factor,
        ]
        results = []

        for weight, prequant_scaling_factor in zip(weights, prequant_scaling_factors):
            results.append(
                resmooth_and_get_scale_and_amax(
                    merged_weights=weight,
                    pre_quant_scales=[prequant_scaling_factor],
                    ranks=1,
                    group_size=block_size,
                    avg_pre_quant_scale=pre_quant_scale,
                    quantization=quantization,
                )
            )

        # Resmooth function returns weight_scaling_factor_2 in NVFP4_AWQ case and weight_amax otherwise.
        q_weight, q_weight_scaling_factor, _ = results[0]
        k_weight, k_weight_scaling_factor, _ = results[1]
        v_weight, v_weight_scaling_factor, _ = results[2]

        if quantization == QUANTIZATION_NVFP4_AWQ:
            # In NVFP4_AWQ, scale_or_amax is updated weight_scaling_factor_2
            q_weight_scaling_factor_2 = NVFP4QTensor.get_weights_scaling_factor_2(q_weight)
            k_weight_scaling_factor_2 = NVFP4QTensor.get_weights_scaling_factor_2(k_weight)
            v_weight_scaling_factor_2 = NVFP4QTensor.get_weights_scaling_factor_2(v_weight)

        q_prequant_scaling_factor = k_prequant_scaling_factor = v_prequant_scaling_factor = (
            pre_quant_scale
        )

    config.q = LinearConfig(linear_type=LINEAR_COLUMN)
    config.q.weight = q_weight.type(dtype)
    config.q.bias = q_bias.type(dtype) if q_bias is not None else None
    config.q.activation_scaling_factor = q_activation_scaling_factor
    config.q.weights_scaling_factor = q_weight_scaling_factor
    config.q.weights_scaling_factor_2 = q_weight_scaling_factor_2
    config.q.prequant_scaling_factor = q_prequant_scaling_factor
    config.q.awq_block_size = block_size
    config.q.quantization = q_quantization

    config.k = LinearConfig(linear_type=LINEAR_COLUMN)
    config.k.weight = k_weight.type(dtype)
    config.k.bias = k_bias.type(dtype) if k_bias is not None else None
    config.k.activation_scaling_factor = k_activation_scaling_factor
    config.k.weights_scaling_factor = k_weight_scaling_factor
    config.k.weights_scaling_factor_2 = k_weight_scaling_factor_2
    config.k.prequant_scaling_factor = k_prequant_scaling_factor
    config.k.awq_block_size = block_size
    config.k.quantization = k_quantization

    config.v = LinearConfig(linear_type=LINEAR_COLUMN)
    config.v.weight = v_weight.type(dtype)
    config.v.bias = v_bias.type(dtype) if v_bias is not None else None
    config.v.activation_scaling_factor = v_activation_scaling_factor
    config.v.weights_scaling_factor = v_weight_scaling_factor
    config.v.weights_scaling_factor_2 = v_weight_scaling_factor_2
    config.v.prequant_scaling_factor = v_prequant_scaling_factor
    config.v.awq_block_size = block_size
    config.v.quantization = v_quantization

    if not ext_config.attention_head_size:
        ext_config.attention_head_size = config.q.weight.shape[0] * training_tp // num_heads

    return config


def build_linear_config(module: nn.Module, linear_type: str, dtype: torch.dtype) -> LinearConfig:
    """Builds the linear config for the module."""
    if has_mcore and not isinstance(module, MegatronModule):
        # Check only for HF model, not Mcore model
        assert is_linear(module)

    torch_weight = module.weight

    if "NormHead" in type(module).__name__:
        torch_weight = torch.nn.functional.normalize(torch_weight)
    elif "Conv1D" in type(module).__name__ and not (
        hasattr(module, "input_quantizer") or hasattr(module, "output_quantizer")
    ):
        # Transpose Conv1D weights to linear unless it has been transposed by the quantization.
        torch_weight = torch_weight.T

    weight = torch_weight.type(dtype)

    config = LinearConfig(linear_type=linear_type)
    config.weight = weight

    if hasattr(module, "bias") and module.bias is not None:
        config.bias = module.bias.type(dtype)

    config.activation_scaling_factor = get_activation_scaling_factor(module)
    config.weights_scaling_factor = get_weight_scaling_factor(module)
    config.weights_scaling_factor_2 = get_weight_scaling_factor_2(module)
    config.prequant_scaling_factor = get_prequant_scaling_factor(module, dtype)
    config.awq_block_size = get_weight_block_size(module)
    config.quantization = get_quantization_format(module)
    return config


def build_attention_config(
    module: nn.Module,
    model_metadata_config,
    dtype: torch.dtype,
    ext_config: DecoderLayerConfig = None,
    tp_size: int = 1,
) -> AttentionConfig:
    """Builds the attention config from the module."""
    assert is_attention(module)

    config = AttentionConfig()
    if hasattr(module, "rotary_dim"):
        config.rotary_dim = module.rotary_dim
    if hasattr(module, "clip_qkv"):
        config.clip_qkv = module.clip_qkv

    qkv_modules = []
    q = None
    k = None
    v = None
    for name, layer in module.named_children():
        if is_linear(layer):
            if _is_qkv(name):
                qkv_modules.append(layer)
            elif "q" in name:
                q = layer
            elif "k" in name:
                k = layer
            elif "v" in name:
                v = layer
            else:
                # The dense layer
                config.dense = build_linear_config(layer, LINEAR_ROW, dtype)
        elif is_layernorm(layer):
            if "q" in name.lower():
                config.q_layernorm = build_layernorm_config(layer, dtype)
            elif "k" in name.lower():
                config.k_layernorm = build_layernorm_config(layer, dtype)
            else:
                raise NotImplementedError(f"{name}: {layer} not recognized")
        elif "model_type" in model_metadata_config:
            if model_metadata_config["model_type"] == "t5":
                config.rel_attn_table = RelativeAttentionTableConfig(
                    weight=layer.weight.T.type(dtype)
                )

    if not qkv_modules:
        assert q
        assert k
        assert v
        qkv_modules = [q, k, v]

    config.qkv = build_qkv(qkv_modules, model_metadata_config, dtype, ext_config, tp_size=tp_size)

    config.kv_cache_scaling_factor = get_kv_cache_scaling_factor(qkv_modules)
    if config.kv_cache_scaling_factor is not None:
        config.kv_cache_dtype = get_kv_cache_dtype(qkv_modules)

    return config


def _is_qkv(name) -> bool:
    return all([k in name for k in ["q", "k", "v"]]) or "W_pack" in name or "c_attn" in name


def _get_hidden_act(act_func) -> str:
    """Returns the name of the hidden activation functon based on ACT2FN."""
    if isinstance(act_func, str):
        return act_func

    # Falcon activation, "nn.GELU" is equivalent to "gelu" in ACT2FN
    if isinstance(act_func, nn.GELU):
        return "gelu"

    if hasattr(act_func, "func") and act_func.func == nn.functional.gelu:
        return "gelu"

    for name, func in ACT2FN.items():
        # TRT LLM uses "squared-relu" activation keyword.
        if name == "relu2":
            name = "squared-relu"
        if isinstance(func, tuple):
            if isinstance(act_func, func[0]):
                return name
        elif isinstance(act_func, func):
            return name

    return act_func.__name__


def build_mlp_config(
    module: nn.Module,
    decoder_type,
    dtype: torch.dtype,
    hidden_act: Optional[str] = None,
) -> MLPConfig:
    """Builds the MLP config for the module."""
    assert is_mlp(module)

    config = MLPConfig()
    # fc1 and gate will be merged to the same layer for chatglm
    if decoder_type == "chatglm":
        config.merged_fc1_gate = True

    def _split_gate_from_fc(decoder_type, module, fc_name, fc_layer):
        if (
            "ColumnParallelLinear" in type(fc_layer).__name__
            and hasattr(module.config, "gated_linear_unit")
            and module.config.gated_linear_unit
        ):
            return True

        if decoder_type == "chatglm":
            return True

        if decoder_type != "gpt":
            return False

        if "dense_h_to_4h" in fc_name and "dense_h_to_4h_2" not in fc_name:
            return True

        return False

    # TODO: We may want to refactor these keywords/model mapping
    fc_keywords = set(
        [
            "c_fc",  # gpt2
            "fc_in",  # gptj
            "gate_proj",  # llama, baichuan, recurrentgemma
            "dense_h_to_4h",  # falcon, chatglm, bloom
            "linear_fc1",
            "w2",  # qwen
            "fc1",  # phi, gemma
            "gate_up_proj",  # phi
            "wi_0",  # t5
            "wi",  # t5
            "c_fc_0",  # exaone
        ]
    )
    proj_keywords = set(
        [
            "c_proj",  # gpt2, qwen, exaone
            "fc_out",  # gptj
            "dense_4h_to_h",  # falcon, chatglm, bloom
            "4h_to_h",
            "down_proj",  # llama, baichuan, mpt, phi, recurrentgemma, nemotron
            "linear_fc2",
            "proj",
            "fc2",  # phi, gemma
            "wo",  # t5
        ]
    )
    gate_keywords = set(
        [
            "up_proj",  # llama, baichuan, recurrentgemma
            "dense_h_to_4h_2",
            "w1",  # qwen
            "wi_1",  # t5
            "c_fc_1",  # exaone
        ]
    )

    for name, layer in module.named_children():
        if is_linear(layer):
            # Arctic (llama-based MoE, decoder_type is "llama") has MLP keyword conflicts with Qwen
            # Arctic's residual MLP use w1 for fc, w2 for proj, w3 for gate
            if type(module).__name__ in ["ArcticMLP", "InternLM2MLP"]:
                fc_keywords.discard("w2")
                gate_keywords.discard("w1")
                fc_keywords.add("w1")
                proj_keywords.add("w2")
                gate_keywords.add("w3")

            if decoder_type == "mpt":
                fc_keywords.add("up_proj")
                gate_keywords.discard("up_proj")

            if type(module).__name__ in [
                "TLGv4MLP",
                "Phi3SmallMLP",
                "NemotronMLP",
            ]:  # for TLGv4ForCausalLM
                fc_keywords.add("up_proj")
                gate_keywords.discard("up_proj")

            split_gate = _split_gate_from_fc(decoder_type, module, name, layer)

            if any([keyword == name for keyword in fc_keywords]):
                # swap fused fc and gate
                if decoder_type in ["chatglm", "phi3"]:
                    weights = torch.chunk(layer.weight.type(dtype), 2, dim=0)
                    weights = (weights[1], weights[0])
                    layer.weight.data = torch.cat(weights, dim=0)
                    if hasattr(layer, "weight_quantizer") and hasattr(
                        layer.weight_quantizer, "amax"
                    ):
                        merged_amax = layer.weight_quantizer.amax
                        if merged_amax is not None and merged_amax.numel() != 1:
                            amax_chunks = torch.chunk(merged_amax, 2, dim=0)
                            layer.weight_quantizer.amax = torch.cat(
                                [amax_chunks[1], amax_chunks[0]], dim=0
                            )

                if split_gate:
                    # We have to split the gate from the fc
                    weights = torch.chunk(layer.weight.type(dtype), 2, dim=0)
                    activation_scaling_factor = get_activation_scaling_factor(layer)
                    weight_scaling_factor = get_weight_scaling_factor(layer)
                    weight_scaling_factor_2 = get_weight_scaling_factor_2(layer)
                    prequant_scaling_factor = get_prequant_scaling_factor(layer, dtype)

                    weight_scaling_factors = [None, None]

                    if weight_scaling_factor is not None:
                        if weight_scaling_factor.numel() != 1:
                            # for Int8 SQ case, we split the weight scaling factor into two parts.
                            weight_scaling_factors = torch.chunk(weight_scaling_factor, 2, dim=0)
                        else:
                            # for FP8 case that weight_scaling_factor is a scalar, we repeat it for the gate.
                            weight_scaling_factors = (
                                [weight_scaling_factor] * 2
                                if weight_scaling_factor is not None
                                else [None, None]
                            )

                    config.fc = LinearConfig()
                    config.fc.linear_type = LINEAR_COLUMN
                    config.fc.weight = weights[0]
                    config.fc.weights_scaling_factor = weight_scaling_factors[0]
                    config.fc.weights_scaling_factor_2 = weight_scaling_factor_2
                    config.fc.activation_scaling_factor = activation_scaling_factor
                    config.fc.prequant_scaling_factor = prequant_scaling_factor
                    config.fc.awq_block_size = get_weight_block_size(layer)
                    config.fc.quantization = get_quantization_format(layer)
                    config.gate = LinearConfig()

                    config.gate.linear_type = LINEAR_COLUMN
                    config.gate.weight = weights[1]
                    config.gate.weights_scaling_factor = weight_scaling_factors[1]
                    config.gate.weights_scaling_factor_2 = weight_scaling_factor_2
                    config.gate.activation_scaling_factor = activation_scaling_factor
                    config.gate.prequant_scaling_factor = prequant_scaling_factor
                    config.gate.awq_block_size = get_weight_block_size(layer)
                    config.gate.quantization = get_quantization_format(layer)
                else:
                    config.fc = build_linear_config(layer, LINEAR_COLUMN, dtype)

            elif any([keyword == name for keyword in proj_keywords]):
                config.proj = build_linear_config(layer, LINEAR_ROW, dtype)
            elif any([keyword == name for keyword in gate_keywords]):
                config.gate = build_linear_config(layer, LINEAR_COLUMN, dtype)

    assert config.proj is not None and config.fc is not None, "proj or fc can not be found"

    # Override hidden_act based on decoder_type
    if decoder_type in ["bloom", "glm"]:
        hidden_act = "gelu"
    if decoder_type == "phi3":
        hidden_act = "swiglu"

    if hidden_act is None:
        if hasattr(module, "activation"):
            hidden_act = module.activation
        elif hasattr(module, "activation_func"):
            # MCore activation_func can be swiglu (gated silu) or squared_relu.
            hidden_act = module.activation_func.__name__.replace("_", "-")
            if hidden_act in ["glu", "silu"]:
                hidden_act = "swiglu" if decoder_type == "gpt" else "silu"
        else:
            for act in ["act", "act_fn", "activation_fn"]:
                if hasattr(module, act):
                    hidden_act = _get_hidden_act(getattr(module, act)).split("_")[0]
                    break

        if hidden_act is None:
            raise NotImplementedError(f"{module} not supported.")

    config.hidden_act = hidden_act
    return config


def _get_mixtral_expert(experts: nn.Module, export_id: int, linear_name: str):
    # Mixtral experts layout is:
    # experts[0]:
    #   w1
    #   w2
    #   w3
    # experts[1]:
    #   w1
    #   w2
    #   w3
    # ...
    return getattr(experts[export_id], linear_name)


def _get_dbrx_expert(experts: nn.Module, export_id: int, linear_name: str):
    # DBRX experts layout is:
    # experts:
    #   w1[0]
    #   w1[1]
    #   ...
    #   w2[0]
    #   w2[1]
    #   ...
    #   v1[0]
    #   v1[1]
    #   ...
    return getattr(experts, linear_name)[export_id]


def _build_stacked_linear(
    experts: nn.Module, module_name, linear_type, dtype, num_experts, expert_getter
):
    config = LinearConfig(linear_type=linear_type)

    # weights
    config.weight = torch.stack(
        [expert_getter(experts, i, module_name).weight.type(dtype) for i in range(num_experts)]
    )

    # bias
    first_module = expert_getter(experts, 0, module_name)
    if hasattr(first_module, "bias") and first_module.bias is not None:
        raise ValueError("Unexpected bias tensors inside MOE modules.")

    # scaling factors
    def get_stacked_scaling_factors(experts, get_function, module_name, karg={}):
        expert_0_scaling_factor = get_function(expert_getter(experts, 0, module_name), **karg)

        if expert_0_scaling_factor is None:
            return None

        dtype = expert_0_scaling_factor.dtype
        scaling_factors = [
            get_function(expert_getter(experts, i, module_name), **karg) for i in range(num_experts)
        ]

        if dtype == torch.float8_e4m3fn:
            scaling_factors = [sf.view(torch.uint8) for sf in scaling_factors]
            return torch.stack(scaling_factors).view(torch.float8_e4m3fn)

        return torch.stack(scaling_factors)

    config.activation_scaling_factor = get_stacked_scaling_factors(
        experts, get_activation_scaling_factor, module_name
    )
    # The moe plugin only supports a single activation scaling factor for all experts
    if config.activation_scaling_factor is not None:
        config.activation_scaling_factor = config.activation_scaling_factor.max().unsqueeze(0)
    config.weights_scaling_factor = get_stacked_scaling_factors(
        experts, get_weight_scaling_factor, module_name
    )
    config.weights_scaling_factor_2 = get_stacked_scaling_factors(
        experts, get_weight_scaling_factor_2, module_name
    )
    config.prequant_scaling_factor = get_stacked_scaling_factors(
        experts, get_prequant_scaling_factor, module_name, {"dtype": dtype}
    )
    config.awq_block_size = get_weight_block_size(expert_getter(experts, 0, module_name))
    config.quantization = get_quantization_format(experts)

    return config


def build_stacked_experts(
    experts: nn.Module,
    dtype: torch.dtype,
    linear_names: List[str],
    num_experts,
    expert_getter,
):
    """Builds the experts_weight_1 and experts_weight_2 configs for the experts."""
    experts_weight_1 = _build_stacked_linear(
        experts, linear_names[0], LINEAR_COLUMN, dtype, num_experts, expert_getter
    )
    experts_weight_2 = _build_stacked_linear(
        experts, linear_names[1], LINEAR_ROW, dtype, num_experts, expert_getter
    )
    if len(linear_names) > 2:
        # Only for HF model, as Mcore model only has two fc layers in MoE
        experts_weight_3 = _build_stacked_linear(
            experts, linear_names[2], LINEAR_COLUMN, dtype, num_experts, expert_getter
        )

        # Concat w1 and w3 into w1
        experts_weight_1.weight = torch.concat(
            [experts_weight_3.weight, experts_weight_1.weight], dim=-2
        )

    def _resmooth_stack_weights(experts_list):
        """Resmooth stacked weights for experts for int4_awq."""
        resmooth_weights = []
        resmooth_weights_scaling_factors = []
        # List holds resmoothed weight_scaling_factor_2 for NVFP4_AWQ
        resmooth_weight_scaling_factors_2 = []
        group_size = experts_list[-1].awq_block_size
        quantization = experts_list[-1].quantization

        # Get avg pre_quant_scale amongst experts
        experts_avg_pre_quant_scale = [
            torch.mean(expert.prequant_scaling_factor, dim=0) for expert in experts_list
        ]
        # Get merged prequant_scale if more than one experts linear layer is provided in the list
        merged_avg_pre_quant_scale = torch.mean(torch.stack(experts_avg_pre_quant_scale), dim=0)

        # resmooth each expert as w1 and w3 will share the average prequant_scaling_factor
        for idx in range(experts_list[-1].weight.shape[0]):
            # weights should be concated already in the last entry of experts_list
            merged_weight = experts_list[-1].weight[idx]
            pre_quant_scales = [expert.prequant_scaling_factor[idx] for expert in experts_list]
            (
                resmooth_weight,
                resmooth_scaling_factor,
                _,
            ) = resmooth_and_get_scale_and_amax(
                merged_weight,
                pre_quant_scales,
                len(pre_quant_scales),
                group_size,
                merged_avg_pre_quant_scale,
                quantization,
            )
            resmooth_weights.append(resmooth_weight)
            resmooth_weights_scaling_factors.append(resmooth_scaling_factor)
            if quantization == QUANTIZATION_NVFP4_AWQ:
                resmooth_weight_scaling_factors_2.append(
                    NVFP4QTensor.get_weights_scaling_factor_2(resmooth_weight)
                )

        # We view torch.float8_e4m3fn as uint8 to enable stack operation
        if quantization == QUANTIZATION_NVFP4_AWQ:
            stacked_resmooth_weights_scaling_factors = torch.stack(
                [
                    resmooth_scaling_factor.view(torch.uint8)
                    for resmooth_scaling_factor in resmooth_weights_scaling_factors
                ]
            ).view(torch.float8_e4m3fn)
        else:
            stacked_resmooth_weights_scaling_factors = torch.stack(resmooth_weights_scaling_factors)

        return (
            torch.stack(resmooth_weights),
            stacked_resmooth_weights_scaling_factors,
            merged_avg_pre_quant_scale,
            torch.stack(resmooth_weight_scaling_factors_2)
            if quantization == QUANTIZATION_NVFP4_AWQ
            else None,
        )

    def _max_stack_scaling_factor(weights_scaling_factor_1, weights_scaling_factor_3):
        return torch.stack([weights_scaling_factor_3, weights_scaling_factor_1]).max(dim=0).values

    # scaling factors
    if experts_weight_1.weights_scaling_factor_2 is not None:
        if experts_weight_1.weights_scaling_factor_2.dim() in [1, 2]:
            experts_weight_1.weights_scaling_factor_2 = _max_stack_scaling_factor(
                experts_weight_3.weights_scaling_factor_2,
                experts_weight_1.weights_scaling_factor_2,
            )
        else:
            raise NotImplementedError(
                "Unexpected shape of weights_scaling_factor_2. The quantization algorithm is not"
                " supported."
            )
    # TODO: check if this works with int8_sq
    if experts_weight_1.weights_scaling_factor is not None:
        # fp8 case, per-tensor quantization
        if experts_weight_1.weights_scaling_factor.dim() == 2:
            if len(linear_names) > 2:
                # For HF model
                experts_weight_1.weights_scaling_factor = _max_stack_scaling_factor(
                    experts_weight_3.weights_scaling_factor,
                    experts_weight_1.weights_scaling_factor,
                )
            else:
                # For Mcore model, as Mcore model only has two fc layers in MoE
                experts_weight_1.weights_scaling_factor = _max_stack_scaling_factor(
                    experts_weight_1.weights_scaling_factor,
                    experts_weight_1.weights_scaling_factor,
                )
        # nvfp4 case, weights_scaling_factor need to be requantized with updated weights_scaling_factor_2
        elif experts_weight_1.quantization == QUANTIZATION_NVFP4:
            # Requantize weights_scaling_factor with new weight_scaling_factor_2 (max of weight_scaling_factor_2's)
            experts_weight_1.weights_scaling_factor = NVFP4QTensor.get_weights_scaling_factor(
                experts_weight_1.weight,
                experts_weight_1.awq_block_size,
                experts_weight_1.weights_scaling_factor_2.view(-1, 1, 1),
            )
        elif experts_weight_1.quantization == QUANTIZATION_NVFP4_AWQ:
            # Resmooth weights using new pre_quant_scale
            (
                experts_weight_2.weight,
                experts_weight_2.weights_scaling_factor,
                experts_weight_2.prequant_scaling_factor,
                experts_weight_2.weights_scaling_factor_2,
            ) = _resmooth_stack_weights([experts_weight_2])

            (
                experts_weight_1.weight,
                _,
                experts_weight_1.prequant_scaling_factor,
                experts_weight_1.weights_scaling_factor_2,
            ) = _resmooth_stack_weights([experts_weight_3, experts_weight_1])
            # Requantize weight_scaling_factor
            experts_weight_1.weights_scaling_factor = NVFP4QTensor.get_weights_scaling_factor(
                experts_weight_1.weight,
                experts_weight_1.awq_block_size,
                experts_weight_1.weights_scaling_factor_2.view(-1, 1, 1),
            )

        # int4 awq case, group_wise quantization, and per tensor quantization for prequant_scaling_factor
        elif (
            experts_weight_1.weights_scaling_factor.dim() == 3
            and experts_weight_1.awq_block_size > 0
            and experts_weight_1.prequant_scaling_factor.dim() == 2
        ):
            (
                experts_weight_1.weight,
                experts_weight_1.weights_scaling_factor,
                experts_weight_1.prequant_scaling_factor,
                _,
            ) = _resmooth_stack_weights([experts_weight_3, experts_weight_1])
        else:
            raise NotImplementedError(
                "Unexpected shape of weights_scaling_factor. The quantization algorithm is not"
                " supported."
            )

    # Explicitly move weight to CPU to reduce GPU memory requirement.
    experts_weight_1.weight = experts_weight_1.weight.cpu()
    experts_weight_2.weight = experts_weight_2.weight.cpu()
    return experts_weight_1, experts_weight_2


def build_moe_config(module: nn.Module, decoder_type, dtype: torch.dtype) -> MOEConfig:
    """Builds the MOE config for the module."""
    assert is_moe(module)
    assert decoder_type in ["llama", "dbrx", "phi3"]

    config = MOEConfig()

    # Router: TRT-LLM uses fp32 for router to keep precision
    if decoder_type in ["llama", "phi3"]:
        if has_mcore and isinstance(module, MegatronModule):
            # For Mcore model
            config.router = build_linear_config(module.router, LINEAR_ROW, torch.float32)
        else:
            # For huggingface model
            config.router = build_linear_config(module.gate, LINEAR_ROW, torch.float32)
    elif decoder_type == "dbrx":
        config.router = build_linear_config(module.router.layer, LINEAR_ROW, torch.float32)
    else:
        raise NotImplementedError(f"{decoder_type} not supported")

    # Experts
    experts = ExpertConfig()
    if decoder_type in ["llama", "phi3"]:
        if has_mcore and isinstance(module, MegatronModule):
            # For Mcore model
            experts.fc, experts.proj = build_stacked_experts(
                module.experts.local_experts,
                dtype,
                ["linear_fc1", "linear_fc2"],
                len(module.experts.local_experts),
                _get_mixtral_expert,
            )
            # For Mcore model, experts.fc.weight needs to be flipped along axis = 1
            mid_point = experts.fc.weight.shape[1] // 2
            experts.fc.weight = torch.cat(
                [
                    experts.fc.weight[:, mid_point:, :],
                    experts.fc.weight[:, :mid_point, :],
                ],
                dim=1,
            )
        else:
            # For Huggingface model
            experts.fc, experts.proj = build_stacked_experts(
                module.experts,
                dtype,
                ["w1", "w2", "w3"],
                len(module.experts),
                _get_mixtral_expert,
            )
    elif decoder_type == "dbrx":
        experts.fc, experts.proj = build_stacked_experts(
            module.experts.mlp,
            dtype,
            ["w1_linear", "w2_linear", "v1_linear"],
            len(module.experts.mlp.w1_linear),
            _get_dbrx_expert,
        )
    else:
        raise NotImplementedError(f"{decoder_type} not supported")

    config.experts = experts

    # activation for mixtral and dbrx
    config.hidden_act = "swiglu"

    return config


def build_conv_config(module: nn.Module, dtype: torch.dtype) -> ConvConfig:
    """Builds the conv config for this module."""
    return ConvConfig(
        weight=module.weight.unsqueeze(dim=-1).type(dtype),
        bias=module.bias.type(dtype),
    )


def is_recurrent(module: nn.Module) -> bool:
    """Returns whether the module is a recurrent layer."""
    module_name = type(module).__name__
    return "RecurrentBlock" in module_name


def build_recurrent_config(module: nn.Module, dtype: torch.dtype):
    """Builds the recurrent config for this module."""
    assert is_recurrent(module)

    config = RecurrentConfig()
    config.linear_y = build_linear_config(module.linear_y, linear_type=LINEAR_COLUMN, dtype=dtype)
    # Separate bias from linear_y to y_bias as guided by TRT LLM.
    config.y_bias = config.linear_y.bias
    config.linear_y.bias = None

    config.linear_x = build_linear_config(module.linear_x, linear_type=LINEAR_COLUMN, dtype=dtype)
    config.linear_out = build_linear_config(module.linear_out, linear_type=LINEAR_ROW, dtype=dtype)
    config.conv1d = build_conv_config(module.conv_1d, dtype=dtype)
    config.rg_lru = RgLruConfig()

    config.rg_lru.input_gate = LinearConfig(
        linear_type=LINEAR_GROUP,
        weight=module.rg_lru.input_gate_weight.type(dtype),
        bias=module.rg_lru.input_gate_bias.type(dtype),
        quantization=get_quantization_format(module),
    )

    config.rg_lru.recurrent_gate = LinearConfig(
        linear_type=LINEAR_GROUP,
        weight=module.rg_lru.recurrent_gate_weight.type(dtype),
        bias=module.rg_lru.recurrent_gate_bias.type(dtype),
        quantization=get_quantization_format(module),
    )

    config.rg_lru.recurrent_param = module.rg_lru.recurrent_param.type(dtype)
    return config


def _set_layer_config_from_metaconfig(layer_config, metaconfig):
    for keys, name in HF_CONFIG_MAP:
        for k in keys:
            if k in metaconfig:
                setattr(layer_config, name, metaconfig[k])
    for keys, name in MCORE_CONFIG_MAP:
        for k in keys:
            if k in metaconfig:
                setattr(layer_config, name, metaconfig[k])

    # 2048 is the default TRT LLM max_position_embeddings
    if layer_config.max_position_embeddings == 0:
        layer_config.max_position_embeddings = 2048

    for k in ["_name_or_path"]:
        if k in metaconfig:
            model_path_name = metaconfig[k].replace("-", "_")
            chatglm_version = ""

            for n in ["glm_4", "chatglm3", "chatglm2", "chatglm"]:
                if n in model_path_name:
                    chatglm_version = n.replace("_", "")  # glm_4 -> glm
                    break

            layer_config.chatglm_version = chatglm_version

    # For Falcon variants do not provide new_decoder_architecture, but we can infer it from the model_type
    if "model_type" in metaconfig:
        if metaconfig["model_type"] == "RefinedWeb":
            # Case 1. Falcon-40B / Falcon-40B-instruct
            # https://huggingface.co/tiiuae/falcon-40b/blob/main/layer_config.json
            layer_config.new_decoder_architecture = True
        elif metaconfig["model_type"] == "RefinedWebModel":
            # Case 2. Falcon-7B / Falcon-7B-instruct
            # https://huggingface.co/tiiuae/falcon-7b/blob/main/layer_config.json
            layer_config.new_decoder_architecture = False

    # For Falcon variants, they might not specify the number of kv heads with MQA models, e.g., 7b
    if (
        not layer_config.new_decoder_architecture
        and "multi_query" in metaconfig
        and metaconfig["multi_query"]
    ):
        layer_config.num_kv_heads = 1

    # For Phi3
    for k in ["rope_scaling"]:
        if k in metaconfig and metaconfig[k] is not None:
            for k2 in ["short_factor"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_scaling_short_factors = metaconfig[k][k2]
            for k2 in ["long_factor"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_scaling_long_factors = metaconfig[k][k2]
            # For Phi3 small
            for k2 in ["short_mscale"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_short_mscale = metaconfig[k][k2]
            for k2 in ["long_mscale"]:
                if k2 in metaconfig[k]:
                    layer_config.longrope_long_mscale = metaconfig[k][k2]


def build_decoder_config(
    module: nn.Module,
    model_metadata_config,
    decoder_type: str,
    dtype: torch.dtype,
    tp_size: int = 1,
) -> DecoderLayerConfig:
    """Builds the full decoder config from the module."""
    quantization = get_quantization_format(module)
    config = DecoderLayerConfig(decoder_type=decoder_type, quantization=quantization)
    # Supporting different attention layer config in MCoreGPTModel. If per layer config
    # exists, override the global config.
    if hasattr(module, "self_attention"):
        if hasattr(module.self_attention, "config"):
            if hasattr(module.self_attention.config, "num_attention_heads"):
                config.num_attention_heads = module.self_attention.config.num_attention_heads
            if hasattr(module.self_attention.config, "kv_channels"):
                config.attention_head_size = module.self_attention.config.kv_channels
            if hasattr(module.self_attention.config, "num_query_groups"):
                config.num_kv_heads = module.self_attention.config.num_query_groups

    # Set all config fields in modelopt from HF config
    _set_layer_config_from_metaconfig(config, model_metadata_config)

    if type(module).__name__ == "DbrxBlock":
        # Flatten DBRX attention and ffn
        module_layers = {}
        module_layers.update(dict(getattr(module, "norm_attn_norm").named_children()))
        module_layers.update({"ffn": module.ffn})
    elif decoder_type in ["t5"]:
        # Combine two modules (T5LayerSelfAttention, T5LayerFF) / three modules
        # ((T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF)) of T5 model
        # (depending on whether it's encoder / decoder) into one decoder layer
        combined_module = nn.ModuleList()
        for sub_module in module:
            for layer in sub_module.children():
                combined_module.append(layer)
        module_layers = dict(combined_module.named_children())
    else:
        module_layers = dict(module.named_children())
        if decoder_type in ["exaone"]:
            module_layers.update({"attn": module_layers["attn"].attention})

    for name, layer in module_layers.items():
        # We assume input_layernorm should be before the post_layernorm in decoder block layout,
        # and residual_layernorm could be after post_layernorm
        if is_layernorm(layer):
            layernorm_config = build_layernorm_config(layer, dtype)
            # Special attributes naming for T5 model (Encoder-Decoder model)
            if decoder_type in ["t5"]:
                _update_encoder_decoder_layernorm_config(
                    model_metadata_config, config, layernorm_config
                )
            # For all decoder only models
            else:
                if name in ["ln_mlp"]:
                    config.mlp_layernorm = layernorm_config
                elif config.decoder_type == "gemma2" and "post_attention_layernorm" in name:
                    config.post_layernorm = layernorm_config
                elif config.decoder_type == "gemma2" and "pre_feedforward_layernorm" in name:
                    config.pre_feedforward_layernorm = layernorm_config
                elif config.decoder_type == "gemma2" and "post_feedforward_layernorm" in name:
                    config.post_feedforward_layernorm = layernorm_config
                elif config.input_layernorm is None:
                    config.input_layernorm = layernorm_config
                elif config.post_layernorm is None:
                    config.post_layernorm = layernorm_config
                else:
                    assert model_metadata_config[
                        "parallel_attn_mlp_res"
                    ], "Unexpected layernorm in a layer"
                    config.residual_layernorm = layernorm_config

        elif is_attention(layer):
            # For models where a linear may replace the attention/MLP module (e.g. Deci models)
            if is_linear(layer):
                config.attn_replacing_linear = build_linear_config(
                    layer.linear_attn, "column", dtype
                )
            else:
                if decoder_type in ["bloom", "falcon", "phi3small", "internlm"]:
                    model_metadata_config["head_is_first_dim"] = True
                attention_config = build_attention_config(
                    layer, model_metadata_config, dtype, config, tp_size=tp_size
                )
                # For T5 decoder
                if decoder_type in ["t5"] and model_metadata_config["enc_dec"] == "dec":
                    # We assume self_attention should be before the cross_attention in decoder block layout
                    if config.self_attention is None:
                        config.self_attention = attention_config
                    else:
                        config.cross_attention = attention_config
                elif decoder_type == "mllama":
                    if "cross" in type(layer).__name__.lower():
                        config.cross_attention = attention_config
                    else:
                        config.self_attention = attention_config
                else:
                    config.attention = attention_config

        elif is_moe(layer):
            if quantization not in [QUANTIZATION_NONE, QUANTIZATION_FP8]:
                warn(f"TensorRT-LLM may not support MOE quantization {quantization}")
            config.mlp = build_moe_config(layer, decoder_type, dtype)

        elif is_recurrent(layer):
            config.recurrent = build_recurrent_config(layer, dtype)

        # We assume MoE layer should be before the residual MLP layer in decoder block layout,
        # so MLP layer after a MoE layer is treated as a residual MLP layer
        elif is_mlp(layer):
            if config.mlp is None:
                # For models where a linear may replace the attention/MLP module (e.g. Deci models)
                if is_linear(layer):
                    config.mlp_replacing_linear = build_linear_config(
                        layer.linear_mlp, "column", dtype
                    )
                else:
                    config.mlp = build_mlp_config(
                        layer,
                        decoder_type,
                        dtype,
                        hidden_act=model_metadata_config.get("hidden_act", None),
                    )
            else:
                assert model_metadata_config["parallel_attn_mlp_res"], "Unexpected mlp in a layer"
                config.residual_mlp = build_mlp_config(
                    layer,
                    decoder_type,
                    dtype,
                    hidden_act=model_metadata_config.get("hidden_act", None),
                )

    config.gate_attn = getattr(module, "cross_attn_attn_gate", None)
    config.gate_ffwd = getattr(module, "cross_attn_mlp_gate", None)

    config = _move_input_layernorm_for_noop_attention(config)
    return config


def build_medusa_heads_config(
    model: Optional[nn.Module], dtype: torch.dtype
) -> Optional[List[MedusaHeadConfig]]:
    """Build a list of MedusaHeadConfig if exists.

    Following TensorRT-LLM's Medusa implementation, all Medusa heads (num_medusa_heads) should be
    placed inside a 'torch.nn.ModuleList' with attribute name 'medsua_heads'. A Medusa head composes
    an additional 'lm_head' (vocab_size, hidden_size) and a list (num_medusa_layers) of Medusa layer
    (LinearActConfig). The only supported hidden_act for the layer is 'silu'. All Linear layers are
    column-parallel.
    """
    # medusa_heads: Optional[nn.Module] = None

    def get_medusa_heads(model: nn.Module) -> Optional[nn.Module]:
        """Return the MedusaHead is exists."""
        # MCore GPTModel impl
        if hasattr(model, "medusa_heads"):
            return model.medusa_heads
        return None

    medusa_heads = get_medusa_heads(model)

    if medusa_heads is None:
        return None

    configs = []
    for medusa_head in medusa_heads:
        config = MedusaHeadConfig()
        layer_configs = []
        if isinstance(medusa_head, torch.nn.Sequential):
            # In the HF MedusaModel, nn.Sequencetial is used.
            config.lm_head = build_linear_config(medusa_head[-1], LINEAR_COLUMN, dtype)
            for layer in medusa_head[0:-1]:
                layer_config = LinearActConfig()
                layer_config.linear = build_linear_config(layer.linear, LINEAR_COLUMN, dtype)
                # NOTE: only silu is supported now.
                layer_config.hidden_act = "silu"
                layer_configs.append(layer_config)

        else:
            # In the Megatron, we have MedusaHead define and lm_head is a submodule.
            config.lm_head = build_linear_config(medusa_head.lm_head, LINEAR_COLUMN, dtype)
            for layer in medusa_head.medusa_layers:
                layer_config = LinearActConfig()
                layer_config.linear = build_linear_config(layer.linear, LINEAR_COLUMN, dtype)
                # NOTE: only silu is supported now.
                layer_config.hidden_act = "silu"
                layer_configs.append(layer_config)
        config.medusa_layers = layer_configs
        configs.append(config)
    return configs


def _split_fused_qkv_weight_and_scaling(
    weight: torch.Tensor,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    training_tp: int = 1,
    is_scaling_factor: bool = False,
    keep_channel_order: bool = False,
):
    """Reorder the qkv weight for spliting QKV weights.

    The shape of the fused QKV weights in HF is different from the shape that
    TRT-LLM requires. In particular, the weight of HF consists of interleaved
    q, k, v head weights, while that of TRT-LLM is contigous.
        HF     : [q1, k1, v1, ..., qh, kh, vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, vh]
    where qi, vi, ki are weight vectors corresponding to attention head i.
    It's similar to multi/grouped query attention cases.
    if keep_channel_order
        HF     : [q1, ..., qh, k1, ..., kh, v1, ..., vh]
        TRT-LLM: [q1, ..., qh, k1, ..., kh, v1, ..., vh]
    """
    # Query types and expected kv heads.
    #  - Conventional MHA: num_heads = num_kv_heads
    #  - Multi-Query Attention: num_kv_heads = 1
    #  - Grouped-Query Attention: num_heads % num_kv_heads = 0
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    assert (
        num_heads % num_kv_heads == 0
    ), f"num_heads({num_heads}) must be divisible by num_kv_heads({num_kv_heads}))."

    # The number of attention heads per group: N q head + 1 k head + 1 v head.
    num_group_heads = num_heads // num_kv_heads + 2
    num_kv_heads_single_tp = max(num_kv_heads // training_tp, 1)
    size_per_head = weight.shape[0] // num_kv_heads_single_tp // num_group_heads

    if size_per_head != hidden_size // num_heads:
        warn("Warning: qkv have different hidden size than the input.")

    if is_scaling_factor:
        # For AWQ, weight scaling facotrs will be a 2D array (out_feat, in_feat/group_size)
        # Int8_sq can be regarded as a specific instance where the group_size is equal to in_feat.
        output_ch = num_kv_heads_single_tp * num_group_heads * size_per_head
        weight_size = weight.numel()

        qkv_in = weight_size // output_ch
    else:
        qkv_in = hidden_size

    if keep_channel_order:
        kv_size = num_kv_heads * size_per_head
        q_w = weight[: -2 * kv_size, ...]
        k_w = weight[-2 * kv_size : -1 * kv_size, ...]
        v_w = weight[-1 * kv_size :, ...]

        if is_scaling_factor:
            q_w = q_w.reshape(-1)
            k_w = k_w.reshape(-1)
            v_w = v_w.reshape(-1)
    else:
        # Split Q/K/V weights
        weight = weight.reshape(num_kv_heads_single_tp, num_group_heads, size_per_head, qkv_in)
        q_w = weight[:, :-2, ...]  # (nKV, num_heads // nKV, size_per_head, qkv_in)
        k_w = weight[:, -2:-1, ...]  # (nKV, 1, size_per_head, qkv_in)
        v_w = weight[:, -1:, ...]  # (nKV, 1, size_per_head, qkv_in)

        q_w = q_w.reshape(-1, qkv_in)
        k_w = k_w.reshape(-1, qkv_in)
        v_w = v_w.reshape(-1, qkv_in)

        if not is_scaling_factor:
            q_w = q_w.reshape(-1, qkv_in)
            k_w = k_w.reshape(-1, qkv_in)
            v_w = v_w.reshape(-1, qkv_in)
        else:
            q_w = q_w.reshape(-1)
            k_w = k_w.reshape(-1)
            v_w = v_w.reshape(-1)

    return q_w, k_w, v_w


def _update_encoder_decoder_layernorm_config(model_metadata_config, config, layernorm_config):
    # For encoder of Encoder-Decoder model
    if model_metadata_config["enc_dec"] == "enc":
        if config.attention_layernorm is None:
            config.attention_layernorm = layernorm_config
        else:
            config.mlp_layernorm = layernorm_config
    # For decoder of Encoder-Decoder model
    else:
        if config.self_attention_layernorm is None:
            config.self_attention_layernorm = layernorm_config
        elif config.cross_attention_layernorm is None:
            config.cross_attention_layernorm = layernorm_config
        else:
            config.mlp_layernorm = layernorm_config


def _move_input_layernorm_for_noop_attention(
    complete_decoder_config: DecoderLayerConfig,
) -> DecoderLayerConfig:
    """For models with decoder blocks that skip the attention or recurrent, the only layer norm is AFTER the MLP.

    We build DecoderLayerConfigs with the assumption that there are both norms. This function fixes
    this assumption and moved the input_layernorm to be a post_layernorm if necessary.
    """
    if all(
        config is None
        for config in [
            complete_decoder_config.attention,
            complete_decoder_config.self_attention,
            complete_decoder_config.cross_attention,
            complete_decoder_config.attn_replacing_linear,
            complete_decoder_config.recurrent,
        ]
    ):
        assert (
            complete_decoder_config.post_layernorm is None
        ), "Should not have 2 layer norms with no attention"
        complete_decoder_config.post_layernorm = complete_decoder_config.input_layernorm
        complete_decoder_config.input_layernorm = None

    return complete_decoder_config


def update_experts_avg_prequant_scale(experts: nn.Module):
    """Registers experts_pre_quant_scale attribute of each expert with average pre_quant_scale amongst experts."""
    """In NVFP4_AWQ and INT4_AWQ all the experts share prequant_scaling_factor. """
    experts_linear_names = get_experts_linear_names(experts)
    if "mixtral" in type(experts).__name__.lower():
        get_func = _get_mixtral_expert
        num_experts = len(experts.experts)
        experts = experts.experts
    elif "dbrx" in type(experts).__name__.lower():
        get_func = _get_dbrx_expert
        num_experts = len(getattr(experts.experts.mlp, experts_linear_names[0]))
        experts = experts.experts.mlp
    else:
        raise NotImplementedError("MoE model not supported")

    for linear_name in experts_linear_names:
        # Check if pre_quant_scale exists
        assert hasattr(
            get_func(experts, 0, linear_name).input_quantizer, "pre_quant_scale"
        ), "Layer does not have attribute pre_quant_scale"
        experts_avg_pre_quant_scale = torch.mean(
            torch.stack(
                [
                    get_func(experts, i, linear_name).input_quantizer.pre_quant_scale
                    for i in range(num_experts)
                ]
            ),
            dim=0,
        )
        # Register a new experts_pre_quant_scale attribute of each expert with average pre_quant_scale amongst experts
        for i in range(num_experts):
            get_func(
                experts, i, linear_name
            ).input_quantizer.experts_avg_pre_quant_scale = experts_avg_pre_quant_scale.clone()


def get_experts_linear_names(model: torch.nn.Module):
    """Returns linear layer names based on decoder type for MoE models."""
    if "mixtral" in type(model).__name__.lower():
        return ["w1", "w2", "w3"]
    elif "dbrx" in type(model).__name__.lower():
        return ["w1_linear", "w2_linear", "v1_linear"]
    else:
        raise NotImplementedError("MoE model not supported")
