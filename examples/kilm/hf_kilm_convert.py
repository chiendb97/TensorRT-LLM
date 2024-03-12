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
'''
Convert huggingface KiLM-7B-Chat model to numpy file.
Use https://huggingface.co/Qwen/Qwen-7B-Chat as demo.
'''
import argparse
import configparser
import dataclasses
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as multiprocessing
from smoothquant import capture_activation_range, smooth_gemm, smooth_gemm_mlp
from tqdm import tqdm
from transformers import AutoModelForCausalLM  # transformers-4.10.0-py3
from transformers import AutoTokenizer, GenerationConfig
# for debug
from utils.convert import split_and_save_weight

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.runtime.lora_manager import LoraConfig

now_dir = os.path.dirname(os.path.abspath(__file__))


def create_lora_config(args: argparse.Namespace):
    '''update args based on lora dir
    '''
    hf_modules_to_trtllm_modules = {
        "c_attn": "attn_qkv",
        "c_proj": "attn_dense",
        "w1": "mlp_h_to_4h",
        "c_proj": "mlp_4h_to_h",
        "w2": "mlp_gate"
    }  # lora modules on kilm

    trtllm_modules_to_hf_modules = {
        "attn_qkv": "c_attn",
        "attn_dense": "c_proj",
        "mlp_h_to_4h": "w1",
        "mlp_4h_to_h": "c_proj",
        "mlp_gate": "w2",
    }

    lora_config = LoraConfig.from_hf(args.hf_lora_dir,
                                     hf_modules_to_trtllm_modules,
                                     trtllm_modules_to_hf_modules)

    return lora_config


@dataclasses.dataclass(frozen=False)
class ProgArgs:
    out_dir: str
    in_file: str
    max_input_len: int = 2048
    tensor_parallelism: int = 1
    processes: int = 1
    calibrate_kv_cache: bool = False
    smoothquant: float = None
    model: str = "kilm"
    hf_lora_dir: str = None
    max_lora_rank: int = 0
    storage_type: str = "fp32"
    dataset_file: str = None
    chat_format: str = "raw"
    calib_size: int = 32
    dataset_cache_dir: str = None
    load_model_on_cpu: bool = False
    convert_model_on_cpu: bool = False

    @staticmethod
    def parse(args=None) -> 'ProgArgs':
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('--out-dir',
                            '-o',
                            type=str,
                            help='file name of output directory',
                            required=True)
        parser.add_argument('--in-file',
                            '-i',
                            type=str,
                            help='file name of input checkpoint file',
                            required=True)
        parser.add_argument(
            '--max_input_len',
            type=int,
            help=
            "This should be consistent with the max_input_len you used when building engine.",
            default=2048)
        parser.add_argument('--tensor-parallelism',
                            '-tp',
                            type=int,
                            help='Requested tensor parallelism for inference',
                            default=1)
        parser.add_argument(
            "--processes",
            "-p",
            type=int,
            help=
            "How many processes to spawn for conversion (default: 1). Set it to a lower value to reduce RAM usage.",
            default=1)
        parser.add_argument(
            "--calibrate-kv-cache",
            "-kv",
            action="store_true",
            help=
            "Generate scaling factors for KV cache. Used for storing KV cache in int8."
        )
        parser.add_argument(
            "--smoothquant",
            "-sq",
            type=float,
            default=None,
            help="Set the α parameter (see https://arxiv.org/pdf/2211.10438.pdf)"
            " to Smoothquant the model, and output int8 weights."
            " A good first try is 0.5. Must be in [0, 1]")
        parser.add_argument(
            "--model",
            default="kilm",
            type=str,
            help="Specify GPT variants to convert checkpoints correctly",
            choices=["kilm", "gpt2", "santacoder", "starcoder"])
        parser.add_argument('--hf-lora-dir', type=str, default=None)
        parser.add_argument(
            '--max-lora-rank',
            type=int,
            default=64,
            help='maximum lora rank for different lora modules. '
                 'It is used to compute the workspace size of lora plugin.')
        parser.add_argument("--storage-type",
                            "-t",
                            type=str,
                            default="float16",
                            choices=["float32", "float16", "bfloat16"])
        parser.add_argument("--dataset-file",
                            type=str,
                            default=None,
                            help="dataset file for quantize")
        parser.add_argument("--calib-size",
                            type=int,
                            default=32,
                            help="Number of samples for calibration.")
        parser.add_argument("--chat-format",
                            type=str,
                            default="raw",
                            help="chat format")
        parser.add_argument("--load-model-on-cpu", action="store_true")
        parser.add_argument("--convert-model-on-cpu", action="store_true")
        return ProgArgs(**vars(parser.parse_args(args)))


@torch.no_grad()
def smooth_kilm_model(model, scales, alpha, kilm_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        # if not isinstance(module, KiLMBlock):
        if not str(type(module)).endswith("KiLMBlock'>"):
            continue

        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight,
                               scales[layer_name]["x"],
                               module.ln_1.weight,
                               alpha=alpha)
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=1)[0]

        # attention dense
        layer_name = name + ".attn.c_proj"
        smoother3 = smooth_gemm(
            module.attn.c_proj.weight,
            scales[layer_name]["x"],
            None,
            alpha=alpha,
        )
        kilm_smoother[layer_name] = smoother3.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother3
        scales[layer_name]["w"] = module.attn.c_proj.weight.abs().max(dim=1)[0]

        # mlp w1 / w2, because then use some input hidden_states as input, so we need to smooth it with same scale
        mlp_w1_name = name + ".mlp.w1"
        mlp_w2_name = name + ".mlp.w2"
        smoother2 = smooth_gemm_mlp(module.mlp.w1.weight,
                                    module.mlp.w2.weight,
                                    scales[mlp_w1_name]["x"],
                                    module.ln_2.weight,
                                    alpha=alpha)
        scales[mlp_w1_name]["x"] = scales[mlp_w1_name]["x"] / smoother2
        scales[mlp_w2_name]["x"] = scales[mlp_w2_name]["x"] / smoother2
        scales[mlp_w1_name]["w"] = module.mlp.w1.weight.abs().max(dim=1)[0]
        scales[mlp_w2_name]["w"] = module.mlp.w2.weight.abs().max(dim=1)[0]

        # mlp c_proj
        layer_name = name + ".mlp.c_proj"
        smoother4 = smooth_gemm(module.mlp.c_proj.weight,
                                scales[layer_name]["x"],
                                None,
                                alpha=alpha)
        kilm_smoother[layer_name] = smoother4.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother4
        scales[layer_name]["w"] = module.mlp.c_proj.weight.abs().max(dim=1)[0]


# SantaCoder separates Q projection from KV projection
def concat_qkv_weight_bias(q, hf_key, hf_model):
    kv = hf_model.state_dict()[hf_key.replace("q_attn", "kv_attn")]
    return torch.cat([q, kv], dim=-1)


# StarCoder uses nn.Linear for these following ops whose weight matrix is transposed compared to transformer.Conv1D
def transpose_weights(hf_name, param):
    weight_to_transpose = [
        "attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.w1", "mlp.w2"
    ]
    if any([k in hf_name for k in weight_to_transpose]):
        if len(param.shape) == 2:
            param = param.transpose(0, 1)
    return param


def convert_kilm_name(orig_name):
    global_weights = {
        "transformer.wte.weight": "vocab_embedding.weight",
        "transformer.ln_f.weight": "ln_f.weight",
        "lm_head.weight": "lm_head.weight"
    }

    if orig_name in global_weights:
        return global_weights[orig_name]

    _, _, layer_idx, *weight_name = orig_name.split(".")
    layer_idx = int(layer_idx)
    weight_name = "transformer." + ".".join(weight_name)

    per_layer_weights = {
        "transformer.ln_1.weight": "ln_1.weight",
        "transformer.ln_2.weight": "ln_2.weight",
        "transformer.attn.c_attn.weight": "attention.qkv.weight",
        "transformer.attn.c_attn.bias": "attention.qkv.bias",
        "transformer.attn.c_proj.weight": "attention.dense.weight",
        "transformer.attn.c_proj.bias": "attention.dense.bias",
        "transformer.mlp.w1.weight": "mlp.w1.weight",
        "transformer.mlp.w2.weight": "mlp.w2.weight",
        "transformer.mlp.c_proj.weight": "mlp.c_proj.weight",
    }
    return f"layers.{layer_idx}.{per_layer_weights[weight_name]}"


@torch.no_grad()
def hf_kilm_converter(args: ProgArgs):
    lora_config = create_lora_config(args)

    infer_tp = args.tensor_parallelism
    multi_query_mode = True if args.model in ["santacoder", "starcoder"
                                              ] else False
    saved_dir = Path(args.out_dir) / f"{infer_tp}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    # load position_embedding from rank 0
    if args.load_model_on_cpu:
        model = AutoModelForCausalLM.from_pretrained(
            args.in_file,
            device_map=
            "cpu",  # if you gpu memory is not enough, you can set device_map="cpu"
            trust_remote_code=True,
            torch_dtype=str_dtype_to_torch(args.storage_type),
        ).double()  # if you gpu memory is not enough, you can set .half() to .float()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.in_file,
            device_map=
            "auto",  # if you gpu memory is not enough, you can set device_map="cpu"
            trust_remote_code=True,
            torch_dtype=str_dtype_to_torch(args.storage_type),
        ).half()  # if you gpu memory is not enough, you can set .half() to .float()
    model.generation_config = GenerationConfig.from_pretrained(
        args.in_file, trust_remote_code=True)
    act_range = {}
    kilm_smoother = {}
    if args.smoothquant is not None or args.calibrate_kv_cache:
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false")
        from datasets import load_dataset

        dataset = load_dataset("csv", data_files={'validation': args.dataset_file}, split='validation')
        tokenizer = AutoTokenizer.from_pretrained(
            args.in_file,
            legacy=False,
            padding_side='left',
            trust_remote_code=True,
        )
        chat_format = args.chat_format
        tokenizer.pad_token_id = tokenizer.im_end_id
        # use this prompt to make chat model do summarize
        system_prompt = "You are a helpful assistant."
        act_range = capture_activation_range(
            model,
            tokenizer,
            dataset,
            system_prompt=system_prompt,
            chat_format=chat_format,
            max_input_len=args.max_input_len,
            num_samples=args.calib_size
        )
        if args.smoothquant is not None:
            smooth_kilm_model(model, act_range, args.smoothquant, kilm_smoother)

    config = configparser.ConfigParser()
    config["kilm"] = {}
    for key in vars(args):
        config["kilm"][key] = f"{vars(args)[key]}"
    for k, v in vars(model.config).items():
        config["kilm"][k] = f"{v}"
    config["kilm"]["storage_dtype"] = args.storage_type
    config["kilm"]["multi_query_mode"] = str(multi_query_mode)
    with open(saved_dir / "config.ini", 'w') as configfile:
        config.write(configfile)

    storage_type = str_dtype_to_torch(args.storage_type)

    global_weights = ["vocab_embedding.weight", "ln_f.weight", "lm_head.weight"]

    int8_outputs = None
    if args.calibrate_kv_cache:
        int8_outputs = "kv_cache_only"
    if args.smoothquant is not None:
        int8_outputs = "all"

    starmap_args = []
    for name, param in tqdm(
            model.named_parameters(),
            desc="convert and save",
            total=len(list(model.parameters())),
            ncols=80,
    ):
        if "weight" not in name and "bias" not in name:
            continue
        converted_name = convert_kilm_name(name)

        if args.convert_model_on_cpu:
            param = param.cpu()
        if name.replace(".weight", "") in kilm_smoother.keys():
            smoother = kilm_smoother[name.replace(".weight", "")]
            starmap_arg = (
                0,
                saved_dir,
                infer_tp,
                f"{converted_name}.smoother".replace(".weight", ""),
                smoother,
                storage_type,
                None,
                {
                    "int8_outputs": int8_outputs,
                    "multi_query_mode": multi_query_mode,
                    "local_dim": None,
                },
            )
            if args.processes > 1:
                starmap_args.append(starmap_arg)
            else:
                split_and_save_weight(*starmap_arg)

        param = transpose_weights(name, param)
        if converted_name in global_weights:
            if converted_name == "vocab_embedding.weight":
                if lora_config.is_valid and lora_config.embedding_weight is not None:
                    print("set embedding weight from lora to kilm")
                    param = lora_config.embedding_weight

            if converted_name == "lm_head.weight":
                if lora_config.is_valid and lora_config.lm_head_weight is not None:
                    print("set lm head weight from lora to kilm")
                    param = lora_config.lm_head_weight

            torch_to_numpy(param.to(storage_type).cpu()).tofile(
                saved_dir / f"{converted_name}.bin")
        else:
            if 'q_attn' in name:
                param = concat_qkv_weight_bias(param, name, model)
                converted_name = converted_name.replace("query",
                                                        "query_key_value")
            # Needed by QKV projection weight split. With multi_query_mode one does not simply take
            # out_dim and divide it by 3 to get local_dim because out_dim = local_dim + 2 * head_size
            local_dim = model.transformer.h[
                0].attn.embed_dim if multi_query_mode else None
            starmap_arg = (0, saved_dir, infer_tp, converted_name,
                           param.to(storage_type), storage_type,
                           act_range.get(name.replace(".weight", "")), {
                               "int8_outputs": int8_outputs,
                               "multi_query_mode": multi_query_mode,
                               "local_dim": local_dim
                           })
            if args.processes > 1:
                starmap_args.append(starmap_arg)
            else:
                split_and_save_weight(*starmap_arg)

    if args.processes > 1:
        starmap_args = tqdm(starmap_args, desc="saving weights")
        with multiprocessing.Pool(args.processes) as pool:
            pool.starmap(split_and_save_weight, starmap_args)


def run_conversion(args: ProgArgs):
    print("\n=============== Arguments ===============")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("========================================")
    hf_kilm_converter(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    run_conversion(ProgArgs.parse())
