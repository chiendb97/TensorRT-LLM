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
import argparse
import json
import os

# isort: off
import torch
import tensorrt as trt
# isort: on

from utils import get_engine_name

from PIL import Image
import torchvision.transforms as transforms
import torch

from transformers import AutoModel


import tensorrt_llm
from tensorrt_llm import logger
from tensorrt_llm._utils import trt_dtype_to_torch
from tensorrt_llm.runtime import Session, TensorInfo


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, required=True)
    parser.add_argument('--hf_model_dir', type=str, required=True)
    parser.add_argument('--run_hf_test', action='store_true')
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--debug', action='store_true')

    return parser.parse_args()


def load_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Define the resize transformation (e.g., resize to 448x448)
    resize_transform = transforms.Resize((448, 448))

    # Define transformation to convert to tensor
    to_tensor = transforms.ToTensor()

    # Apply transformations
    resized_image = resize_transform(image)
    tensor_image = to_tensor(resized_image)

    # If you need to add a batch dimension (e.g., for model input)
    tensor_image = tensor_image.unsqueeze(0)  # Shape: [1, C, H, W]

    print(tensor_image.shape)
    return tensor_image


if __name__ == '__main__':
    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    world_size = config['pretrained_config']['mapping']['world_size']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    model_name = config['pretrained_config']['architecture']

    runtime_rank = tensorrt_llm.mpi_rank() if world_size > 1 else 0

    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = get_engine_name(runtime_rank)
    serialize_path = os.path.join(args.engine_dir, serialize_path)

    stream = torch.cuda.current_stream().cuda_stream
    logger.info(f'Loading engine from {serialize_path}')
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine')
    session = Session.from_serialized_engine(engine_buffer)
    if args.debug: session._print_engine_info()

    pixel_values = load_image(args.image_path)
    pixel_values = pixel_values.to(device=torch.device('cuda'), dtype=torch.float32)

    # NOTE: TRT-LLM perform inference
    output_name = "hidden_states"

    inputs = {
        "pixel_values": pixel_values
    }

    output_info = session.infer_shapes([
        TensorInfo("pixel_values", trt.DataType.FLOAT, pixel_values.shape)
    ])

    outputs = {
        t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
        for t in output_info
    }
    assert output_name in outputs, f'{output_name} not found in outputs, check if build.py set output name correctly'

    logger.info(f"Rank{runtime_rank} is running inference...")
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream)
    assert ok, "Runtime execution failed"
    torch.cuda.synchronize()
    res = outputs[output_name]
    if args.debug: logger.info(f"Outputs:{outputs.keys()}")

    # NOTE: load hf model and perform inference as reference (only on rank0)
    if tensorrt_llm.mpi_rank() == 0:
        logger.info(f"Rank{runtime_rank} is generating HF reference...")
        if args.run_hf_test:
            hf_model = AutoModel.from_pretrained(
                args.hf_model_dir, trust_remote_code=True).vision_model.cuda().to(torch.float32).eval()
            pixel_values = pixel_values.to(device=hf_model.device, dtype=torch.float32)
            with torch.no_grad():
                hf_last_hidden_state = hf_model.forward(output_hidden_states=args.debug,
                                              pixel_values=pixel_values)['last_hidden_state']
            torch.cuda.synchronize()

    if tensorrt_llm.mpi_rank() == 0:
        logger.info(f"Rank{runtime_rank} is comparing with HF reference...")
        logger.info(f"Huggingface output: {hf_last_hidden_state}")
        logger.info(f"TensorRT-LLM output: {res}")
        pass
