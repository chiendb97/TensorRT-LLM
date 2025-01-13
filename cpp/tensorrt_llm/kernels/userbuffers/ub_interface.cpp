/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ub_interface.h"
#if ENABLE_MULTI_DEVICE
namespace tensorrt_llm::runtime::ub
{
void ub_initialize(int tp)
{
    UserBufferAllocator::Instance().initialize(tp);
}

bool ub_is_initialized()
{
    return UserBufferAllocator::Instance().is_initialized();
}

void* ub_allocate(int idx, size_t bytes)
{
    return UserBufferAllocator::Instance().allocate(idx, bytes);
}

void ub_deallocate(void* addr)
{
    UserBufferAllocator::Instance().deallocate(addr);
}

UBBuffer ub_get(int idx)
{
    return UserBufferAllocator::Instance().get(idx);
}

communicator* ub_comm()
{
    return UserBufferAllocator::Instance().comm();
}

bool ub_supported()
{
    return true;
}
}; // namespace tensorrt_llm::runtime::ub

namespace tensorrt_llm::kernels::ub
{
using namespace tensorrt_llm::runtime::ub;

void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    allreduce2_userbuff_inplace_impl(handler, offset, elements, dataType, comm, stream);
}

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    return allgather2_userbuff_residual_impl(handler, offset, elements, hidden_size, residual, dataType, comm, stream);
}

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream)
{
    return allreduce2_userbuff_inplace_rmsnorm_quant_impl(handler, offset, out_handler, out_offset, elements,
        hidden_size, beta, gamma, eps, scalefactor, residual_in, residual_out, dataType, comm, stream);
}
} // namespace tensorrt_llm::kernels::ub
#else
namespace tensorrt_llm::runtime::ub
{
void ub_initialize(int tp) {}

bool ub_is_initialized()
{
    return false;
}

void* ub_allocate(int idx, size_t bytes)
{
    return nullptr;
}

void ub_deallocate(void* addr) {}

UBBuffer ub_get(int idx)
{
    return UBBuffer();
}

communicator* ub_comm()
{
    return nullptr;
}

bool ub_supported()
{
    return false;
}
}; // namespace tensorrt_llm::runtime::ub

namespace tensorrt_llm::kernels::ub
{
using namespace tensorrt_llm::runtime::ub;

void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
}

int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream)
{
    return 0;
}

int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream)
{
    return 0;
}
} // namespace tensorrt_llm::kernels::ub
#endif
