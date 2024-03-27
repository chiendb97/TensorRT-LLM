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

#include "tensorrt_llm/runtime/transformerBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/runtimeBuffers.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

TransformerBuffers::TransformerBuffers()
{
    pastKeyValueLengths = nullptr;
    attentionMask = nullptr;
    positionIds = nullptr;

    presentKeysVals.clear();
    presentKeysValsAlt.clear();
    kvCacheBlockPointersHost = nullptr;
    kvCacheBlockPointersDevice = nullptr;
}

TransformerBuffers::TransformerBuffers(
    TllmRuntime const& runtime, runtime::GptModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(modelConfig.isTransformerBased());
    auto& manager = runtime.getBufferManager();
    auto& engine = runtime.getEngine();

    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    nvinfer1::DataType kvDtype;
    if (modelConfig.usePagedKvCache())
    {
        kvDtype = modelConfig.getKvDataType();
    }
    else
    {
        kvDtype = modelConfig.getQuantMode().hasFp8KvCache()
            ? nvinfer1::DataType::kFP8
            : engine.getTensorDataType(("present_key_value_" + std::to_string(firstLayerId)).c_str());
    }

    if (modelConfig.usePagedKvCache())
    {
        auto const kvCacheBlockPointersType = engine.getTensorDataType("kv_cache_block_pointers");
        kvCacheBlockPointersHost = manager.emptyTensor(MemoryType::kCPU, kvCacheBlockPointersType);
        kvCacheBlockPointersDevice = manager.emptyTensor(MemoryType::kGPU, kvCacheBlockPointersType);
    }
    else
    {
        presentKeysVals = utils::createBufferVector(runtime, localNbLayers, MemoryType::kGPU, kvDtype);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
        maxAttentionWindows = BufferManager::cpu(ITensor::makeShape({localNbLayers}), nvinfer1::DataType::kINT32);
        sinkTokenLengths = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    }
    else
    {
        presentKeysValsAlt = utils::createBufferVector(runtime, localNbLayers, MemoryType::kGPU, kvDtype);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::reshape(GenerationConfig const& generationConfig, KvCacheManager const* kvCacheManager,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;
    auto const maxInputLength = generationConfig.maxInputLength;
    auto const maxAttentionWindow = generationConfig.maxAttentionWindow;

    auto kvCacheReserve = ITensor::makeShape(
        {batchSize, 2, modelConfig.getNbKvHeads(), maxAttentionWindow, modelConfig.getSizePerHead()});
    auto kvCacheShape
        = ITensor::makeShape({batchSize, 2, modelConfig.getNbKvHeads(), maxInputLength, modelConfig.getSizePerHead()});
    if (modelConfig.usePagedKvCache())
    {
        TLLM_CHECK(kvCacheManager);

        auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());

        auto const maxBlocksPerSeq = kvCacheManager->getMaxBlocksPerSeq();
        // reserve batchSize * beamWidth and resize to batchSize
        auto cacheBlockPointersShape = ITensor::makeShape({localNbLayers, batchSize * beamWidth, 2, maxBlocksPerSeq});
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
        cacheBlockPointersShape.d[1] = batchSize;
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
    }
    else
    {
        utils::reshapeBufferVector(presentKeysVals, kvCacheReserve);
    }

    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());

    if (modelConfig.useGptAttentionPlugin())
    {
        pastKeyValueLengths->reshape(ITensor::makeShape({batchSize}));
        maxAttentionWindows->reshape(ITensor::makeShape({localNbLayers}));
        sinkTokenLengths->reshape(ITensor::makeShape({1}));
    }
    else
    {
        utils::reshapeBufferVector(presentKeysValsAlt, kvCacheReserve);
        // present KV cache tensors will be reshaped by shape inference.
        // reshape to the required shape here to make context batch slicing work correctly.
        utils::reshapeBufferVector(presentKeysVals, kvCacheShape);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::reset(BufferManager& manager) {}

TransformerBuffers TransformerBuffers::sliceTo(
    GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, SizeType offset, SizeType batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TransformerBuffers buffers;
    auto const generationBatchSize = generationConfig.batchSize;
    if (modelConfig.usePagedKvCache())
    {
        auto const& realCacheBlockPointersShape = kvCacheBlockPointersHost->getShape();
        auto const localNbLayers = realCacheBlockPointersShape.d[0];
        auto const maxBlocksPerSeq = realCacheBlockPointersShape.d[3];

        // enable slicing by moving generationBatchSize to first dim
        auto const fakeCacheBlockPointersShape
            = ITensor::makeShape({generationBatchSize, localNbLayers, 2, maxBlocksPerSeq});
        TensorPtr kvCacheBlockPointersHostView{ITensor::view(kvCacheBlockPointersHost, fakeCacheBlockPointersShape)};
        TensorPtr kvCacheBlockPointersDeviceView{
            ITensor::view(kvCacheBlockPointersDevice, fakeCacheBlockPointersShape)};

        // slice and reshape to correct shape
        auto const cacheBlockPointersShape = ITensor::makeShape({localNbLayers, batchSize, 2, maxBlocksPerSeq});
        buffers.kvCacheBlockPointersHost = ITensor::slice(kvCacheBlockPointersHostView, offset, batchSize);
        buffers.kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        buffers.kvCacheBlockPointersDevice = ITensor::slice(kvCacheBlockPointersDeviceView, offset, batchSize);
        buffers.kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
    }
    else
    {
        buffers.presentKeysVals = utils::sliceBufferVector(presentKeysVals, offset, batchSize);
    }

    if (modelConfig.useGptAttentionPlugin())
    {
        buffers.pastKeyValueLengths = ITensor::slice(pastKeyValueLengths, offset, batchSize);
        buffers.maxAttentionWindows = maxAttentionWindows;
        buffers.sinkTokenLengths = sinkTokenLengths;
    }
    else
    {
        buffers.presentKeysValsAlt = utils::sliceBufferVector(presentKeysValsAlt, offset, batchSize);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return buffers;
}

static std::vector<SizeType> getPositionIdsContextPhaseGlm(SizeType const& batchSize, SizeType const& maxInputLength,
    SizeType const* pInputLengths, bool useGptAttentionPlugin, bool usePackedInput)
{
    TLLM_CHECK(pInputLengths != nullptr);

    std::vector<SizeType> positionIdsVec(1, 0);
    if (useGptAttentionPlugin)
    {
        if (usePackedInput)
        {
            std::vector<int> pInputLengthsAcc = std::vector<int>(batchSize + 1, 0);
            for (int i = 0; i < batchSize; ++i)
            {
                pInputLengthsAcc[i + 1] = pInputLengthsAcc[i] + pInputLengths[i];
            }

            auto const size = 1 * 2 * pInputLengthsAcc[batchSize];
            positionIdsVec.resize(size, 0);
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + pInputLengthsAcc[b];
                auto const length = pInputLengths[b];
                std::iota(pIdB, pIdB + length, 0);

                pIdB[length - 1] = length - 2;
                pIdB[length - 1 + pInputLengthsAcc[batchSize]] = 1;
            }
        }
        else
        {
            auto const size = batchSize * 2 * maxInputLength;
            positionIdsVec.resize(size, 0);
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + b * 2 * maxInputLength;
                auto const length = pInputLengths[b];
                std::iota(pIdB, pIdB + length, 0);

                pIdB[length - 1] = length - 2;
                pIdB[length - 1 + maxInputLength] = 1;
            }
        }
    }
    else
    {
        TLLM_THROW("Unsupported model without GPT Attention Plugin");
    }

    return positionIdsVec;
}

void TransformerBuffers::prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds,
    TokenIdType const padId, BufferManager& manager, KvCacheManager const* kvCacheManager, SizeType firstBatchSlotIdx,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& promptTuningTasksHost = runtimeBuffers->promptTuningTasksHost;
    auto& promptTuningParams = runtimeBuffers->promptTuningParams;
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const maxInputLength = generationConfig.maxInputLength;
    auto const& inputShape = inputIds->getShape();

    // get local number of layers.
    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());

    if (modelConfig.useGptAttentionPlugin())
    {
        auto pastKeyValueLengthsPtr = bufferCast<SizeType>(*pastKeyValueLengths);
        TLLM_CHECK(pastKeyValueLengths->getSize() == static_cast<std::size_t>(batchSize));

        auto RequestTypesPtr = bufferCast<int32_t>(*requestTypes);
        TLLM_CHECK(requestTypes->getSize() == static_cast<std::size_t>(batchSize));
        std::fill_n(RequestTypesPtr, batchSize, 0);

        auto maxAttentionWindowsPtr = bufferCast<SizeType>(*maxAttentionWindows);
        std::fill_n(maxAttentionWindowsPtr, localNbLayers, generationConfig.maxAttentionWindow);

        bufferCast<SizeType>(*sinkTokenLengths)[0] = generationConfig.sinkTokenLength;

        auto const contextLengthsHostPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const modelVariant = modelConfig.getModelVariant();

        if (modelVariant == GptModelConfig::ModelVariant::kGpt)
        {
            auto const inputSize = inputIds->getSize();
            std::vector<SizeType> positionIdsVec(inputSize);
            auto begin = std::begin(positionIdsVec);
            for (SizeType i = 0; i < batchSize; ++i)
            {
                auto end = begin + (modelConfig.usePackedInput() ? contextLengthsHostPtr[i] : maxInputLength);
                std::iota(begin, end, 0);
                begin = end;
            }
            positionIds = manager.copyFrom(positionIdsVec, inputShape, MemoryType::kGPU);
        }
        else if (modelVariant == GptModelConfig::ModelVariant::kGlm)
        {
            auto const positionIdsVec = getPositionIdsContextPhaseGlm(batchSize, maxInputLength, contextLengthsHostPtr,
                modelConfig.useGptAttentionPlugin(), modelConfig.usePackedInput());
            if (modelConfig.usePackedInput())
            {
                int num_tokens = (int) positionIdsVec.size() / 2;
                auto const positionIdsShape = ITensor::makeShape({2, num_tokens});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
            else
            {
                auto const positionIdsShape = ITensor::makeShape({batchSize, 2, maxInputLength});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
        }
        else
        {
            TLLM_THROW("Unsupported model variant");
        }

        for (SizeType i = 0; i < batchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i];
        }

        if (modelConfig.usePromptTuning())
        {
            std::vector<SizeType> reqBeamWidths(batchSize, 1);
            std::vector<SizeType> reqPromptLengths;
            for (SizeType i = 0; i < batchSize; ++i)
            {
                reqPromptLengths.push_back(contextLengthsHostPtr[i]);
            }

            // Copy the generationInput tasks to host
            promptTuningTasksHost = manager.copyFrom(*promptTuningParams.tasks, MemoryType::kPINNED);

            // Update the tasks tensor
            promptTuningParams.fillTasksTensor(promptTuningTasksHost, batchSize, batchSize, reqBeamWidths,
                reqPromptLengths, manager, modelConfig.usePackedInput());
        }
    }
    else
    {
        attentionMask = manager.copyFrom(*inputIds, MemoryType::kGPU);
        kernels::invokeBuildAttentionMask(*attentionMask, padId, stream);

        auto attentionMaskHost = manager.copyFrom(*attentionMask, MemoryType::kCPU);
        auto const* attentionMaskData = reinterpret_cast<SizeType const*>(attentionMaskHost->data());
        std::vector<SizeType> positionIdsVec(attentionMask->getSize());
        for (SizeType i = 0; i < batchSize; ++i)
        {
            tc::stl_utils::exclusiveScan(attentionMaskData + i * maxInputLength,
                attentionMaskData + (i + 1) * maxInputLength, std::begin(positionIdsVec) + i * maxInputLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskData[i] == 0)
                positionIdsVec[i] = 1;
        positionIds = manager.copyFrom(positionIdsVec, attentionMask->getShape(), MemoryType::kGPU);
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = hiddenStates->getShape().d[hiddenStates->getShape().nbDims - 1];
        auto const hiddenStatesShape = modelConfig.usePackedInput()
            ? ITensor::makeShape({inputShape.d[0], hiddenSize})
            : ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto constexpr contextBeamWidth = 1;
        kvCacheManager->getBlockPointersOfBatch(
            *kvCacheBlockPointersHost, firstBatchSlotIdx, batchSize, contextBeamWidth);
        manager.copy(*kvCacheBlockPointersHost, *kvCacheBlockPointersDevice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

static std::vector<SizeType> getPositionIdsGenerationPhaseGlm(SizeType const& batchSize, SizeType const& beamSize,
    SizeType const& step, SizeType const* pInputLengths, bool useGptAttentionPlugin, bool usePackedInput)
{
    TLLM_CHECK(pInputLengths != nullptr);

    auto const size = 2 * batchSize * beamSize;
    std::vector<SizeType> positionIdsVec(size, 0);
    if (useGptAttentionPlugin)
    {
        if (usePackedInput)
        {
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + b * beamSize * 2;
                auto const length = pInputLengths[b * beamSize];

                for (SizeType bm = 0; bm < beamSize; ++bm)
                {
                    pIdB[bm * 2 + 0] = length - 2;
                    pIdB[bm * 2 + 1] = step + 2;
                }
            }
        }
        else
        {
            for (SizeType b = 0; b < batchSize; ++b)
            {
                auto* pIdB = positionIdsVec.data() + b * beamSize * 2;
                auto const length = pInputLengths[b * beamSize];

                for (SizeType bm = 0; bm < beamSize; ++bm)
                {
                    pIdB[bm * 2 + 0] = length - 2;
                    pIdB[bm * 2 + 1] = step + 2;
                }
            }
        }
    }
    else
    {
        TLLM_THROW("Unsupported model without GPT Attention Plugin");
    }

    return positionIdsVec;
}

void TransformerBuffers::copyAttentionMasks(
    RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto const batchSize = generationConfig.batchSize;
    auto const maxInputLength = generationConfig.maxInputLength;

    // TODO(rkobus) include tiling
    attentionMask = manager.gpu(ITensor::makeShape({batchSize, maxInputLength}), nvinfer1::DataType::kINT32);

    auto const numContextBatches = static_cast<SizeType>(contextBatches.size());
    auto offset = 0;
    for (auto contextBatchId = 0; contextBatchId < numContextBatches; ++contextBatchId)
    {
        auto& buffers = contextBatches.at(contextBatchId);
        auto contextBatchSize = buffers.generationConfig.batchSize;
        auto attentionMaskSlice = ITensor::slice(attentionMask, offset, contextBatchSize);
        manager.copy(*buffers.transformerBuffers->attentionMask, *attentionMaskSlice);
        offset += contextBatchSize;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& logits = runtimeBuffers->logits;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto const beamWidth = generationConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(beamWidth > 1, "Tiling is only necessary for beam search.");

    // Note: If computeContextLogits is true, the copy/expansion is performed in gatherLastTokenLogits.
    if (worldConfig.isLastPipelineParallelRank() && !modelConfig.computeContextLogits())
    {
        // logits needs beamWidth in second dimension
        auto logitsShape = logits->getShape();
        logitsShape.d[1] *= beamWidth;
        utils::tileBufferReplace(logits, beamWidth, manager);
        logits->reshape(logitsShape);
    }

    utils::tileBufferReplace(contextLengthsDevice, beamWidth, manager);

    if (modelConfig.useGptAttentionPlugin())
    {
        utils::tileCpuBufferReplace(contextLengthsHost, beamWidth, manager);
        utils::tileCpuBufferReplace(pastKeyValueLengths, beamWidth, manager);
    }
    else
    {
        utils::tileBufferReplace(attentionMask, beamWidth, manager);
    }

    if (!modelConfig.usePagedKvCache())
    {
        for (auto& buffer : presentKeysVals)
            utils::tileBufferReplace(buffer, beamWidth, manager);
        for (auto& buffer : presentKeysValsAlt)
            utils::tileBufferReplace(buffer, beamWidth, manager);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::postContextStep(RuntimeBuffers* runtimeBuffers,
    std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& requestTypes = runtimeBuffers->requestTypes;
    auto const batchSize = generationConfig.batchSize;
    auto const beamWidth = generationConfig.beamWidth;

    if (modelConfig.useGptAttentionPlugin())
    {
        requestTypes->reshape(ITensor::makeShape({batchSize * beamWidth}));
        auto hostRequestTypes = bufferCast<int32_t>(*requestTypes);
        std::fill_n(hostRequestTypes, requestTypes->getSize(), 1);
    }
    else
    {
        copyAttentionMasks(runtimeBuffers, contextBuffers, manager);
    }

    // TODO(rkobus) handle this more gracefully
    positionIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (modelConfig.computeContextLogits())
    {
        runtimeBuffers->gatherLastTokenLogits(manager, modelConfig, worldConfig);
    }

    if (beamWidth > 1)
    {
        tile(runtimeBuffers, manager, modelConfig, worldConfig);
    }

    if (modelConfig.useGptAttentionPlugin() && modelConfig.usePagedKvCache())
    {
        auto cacheBlockPointersShape = kvCacheBlockPointersHost->getShape();
        cacheBlockPointersShape.d[1] = batchSize * beamWidth;
        kvCacheBlockPointersHost->reshape(cacheBlockPointersShape);
        kvCacheBlockPointersDevice->reshape(cacheBlockPointersShape);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType const step, BufferManager& manager,
    KvCacheManager* kvCacheManager, SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& generationConfig = runtimeBuffers->generationConfig;
    auto& stream = manager.getStream();
    SizeType const batchSize = generationConfig.batchSize;
    SizeType const beamWidth = generationConfig.beamWidth;
    auto const inputShape = [&modelConfig, batchSize, beamWidth]()
    {
        if (modelConfig.usePackedInput())
        {
            // batch in last dim
            return ITensor::makeShape({batchSize * beamWidth});
        }
        else
        {
            // batch in first dim
            return ITensor::makeShape({batchSize * beamWidth, 1});
        }
    }();
    if (modelConfig.useGptAttentionPlugin())
    {
        auto const contextLengthsHostPtr = bufferCast<SizeType const>(*contextLengthsHost);
        auto const pastKeyValueLengthsPtr = bufferCast<SizeType>(*pastKeyValueLengths);
        auto const tensorBatchSize = static_cast<SizeType>(pastKeyValueLengths->getSize());
        SizeType const srcStride{modelConfig.useGptAttentionPlugin() ? 1 : beamWidth};
        TLLM_CHECK(static_cast<std::size_t>(tensorBatchSize * srcStride) == contextLengthsDevice->getSize());
        for (SizeType i = 0; i < tensorBatchSize; ++i)
        {
            pastKeyValueLengthsPtr[i] = contextLengthsHostPtr[i * srcStride] + step;
        }

        auto const modelVariant = modelConfig.getModelVariant();

        if (modelVariant == GptModelConfig::ModelVariant::kGpt)
        {
            positionIds->reshape(inputShape);
            manager.copy(*contextLengthsDevice, *positionIds);
            kernels::invokeAdd(*positionIds, step, stream);
        }
        else if (modelVariant == GptModelConfig::ModelVariant::kGlm)
        {
            auto const positionIdsVec = getPositionIdsGenerationPhaseGlm(batchSize, beamWidth, step,
                contextLengthsHostPtr, modelConfig.useGptAttentionPlugin(), modelConfig.usePackedInput());
            if (modelConfig.usePackedInput())
            {
                auto const positionIdsShape = ITensor::makeShape({2, batchSize * beamWidth});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
            else
            {
                auto const positionIdsShape = ITensor::makeShape({batchSize * beamWidth, 2, 1});
                positionIds = manager.copyFrom(positionIdsVec, positionIdsShape, MemoryType::kGPU);
            }
        }
        else
        {
            TLLM_THROW("Unsupported model variant");
        }
    }
    else
    {
        auto const& shape = attentionMask->getShape();
        auto const nbInputs = shape.d[0];
        auto const oldLength = shape.d[1];
        auto const newLength = oldLength + 1;
        auto const newShape = ITensor::makeShape({nbInputs, newLength});

        TensorPtr newAttentionMask = manager.gpu(newShape, attentionMask->getDataType());
        kernels::invokeExtendAttentionMask(*newAttentionMask, *attentionMask, stream);
        attentionMask = newAttentionMask;

        auto attentionMaskHost = manager.copyFrom(*attentionMask, MemoryType::kCPU);
        auto const* attentionMaskPtr = bufferCast<SizeType>(*attentionMaskHost);

        // TODO old positionIds could be recovered to avoid scan
        std::vector<SizeType> positionIdsVec(attentionMask->getSize());
        for (SizeType i = 0; i < nbInputs; ++i)
        {
            tc::stl_utils::exclusiveScan(attentionMaskPtr + i * newLength, attentionMaskPtr + (i + 1) * newLength,
                std::begin(positionIdsVec) + i * newLength, 0);
        }
        for (std::size_t i = 0; i < positionIdsVec.size(); ++i)
            if (attentionMaskPtr[i] == 0)
                positionIdsVec[i] = 1;
        std::vector<SizeType> positionIdsEndVec(nbInputs);
        for (SizeType i = 0; i < nbInputs; ++i)
            positionIdsEndVec[i] = positionIdsVec[(i + 1) * newLength - 1];

        positionIds = manager.copyFrom(positionIdsEndVec, ITensor::makeShape({nbInputs, 1}), MemoryType::kGPU);
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = hiddenStates->getShape().d[hiddenStates->getShape().nbDims - 1];
        auto const hiddenStatesShape = modelConfig.usePackedInput()
            ? ITensor::makeShape({inputShape.d[0], hiddenSize})
            : ITensor::makeShape({inputShape.d[0], inputShape.d[1], hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (modelConfig.usePagedKvCache())
    {
        for (auto batchIdx = firstBatchSlotIdx; batchIdx < firstBatchSlotIdx + batchSize; ++batchIdx)
        {
            kvCacheManager->addToken(batchIdx);
        }
        kvCacheManager->getBlockPointersOfBatch(*kvCacheBlockPointersHost, firstBatchSlotIdx, batchSize, beamWidth);
        manager.copy(*kvCacheBlockPointersHost, *kvCacheBlockPointersDevice);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void TransformerBuffers::getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers,
    TensorMap& outputBuffers, SizeType const step, TensorPtr const& inputIds, TensorPtr const& commPtrs,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    inputBuffers.clear();
    outputBuffers.clear();

    auto& logits = runtimeBuffers->logits;
    auto& hiddenStates = runtimeBuffers->hiddenStates;
    auto& contextLengthsDevice = runtimeBuffers->contextLengthsDevice;
    auto& contextLengthsHost = runtimeBuffers->contextLengthsHost;
    auto& lastTokenIds = runtimeBuffers->lastTokenIds;
    auto& requestTypes = runtimeBuffers->requestTypes;

    if (worldConfig.isLastPipelineParallelRank())
    {
        // feed a view to TensorRT runtime so reshaping does not change logits buffer
        outputBuffers.insert_or_assign("logits", ITensor::view(logits));
    }
    else
    {
        outputBuffers.insert_or_assign("hidden_states_output", hiddenStates);
    }

    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputBuffers.insert_or_assign("input_ids", inputIds);
    }
    else
    {
        inputBuffers.insert_or_assign("hidden_states_input", hiddenStates);
    }

    inputBuffers.insert_or_assign("context_lengths", contextLengthsDevice);
    if (!modelConfig.computeContextLogits())
    {
        inputBuffers.insert_or_assign("last_token_ids", lastTokenIds);
    }
    inputBuffers.insert_or_assign("position_ids", positionIds);

    auto const localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    if (modelConfig.useGptAttentionPlugin())
    {
        inputBuffers.insert_or_assign("cache_indirection", runtimeBuffers->cacheIndirectionDecoderOutput);
        inputBuffers.insert_or_assign("host_past_key_value_lengths", pastKeyValueLengths);
        inputBuffers.insert_or_assign("host_request_types", requestTypes);
        inputBuffers.insert_or_assign("sequence_length", runtimeBuffers->sequenceLengths);
        inputBuffers.insert_or_assign("host_sink_token_length", sinkTokenLengths);
        inputBuffers.insert_or_assign("host_max_attention_window_sizes", maxAttentionWindows);

        if (modelConfig.usePackedInput())
        {
            inputBuffers.insert_or_assign("host_context_lengths", contextLengthsHost);
        }
        if (modelConfig.usePagedKvCache())
        {
            inputBuffers.insert_or_assign("kv_cache_block_pointers", kvCacheBlockPointersDevice);
            inputBuffers.insert_or_assign("host_kv_cache_block_pointers", kvCacheBlockPointersHost);
        }
        else
        {
            utils::insertTensorVector(inputBuffers, "past_key_value_", presentKeysVals, firstLayerId);
            utils::insertTensorVector(outputBuffers, "present_key_value_", presentKeysVals, firstLayerId);
        }
    }
    else
    {
        inputBuffers.insert_or_assign("attention_mask", attentionMask);
        inputBuffers.insert_or_assign("cache_indirection", runtimeBuffers->cacheIndirectionDecoderOutput);
        utils::insertTensorVector(
            outputBuffers, "present_key_value_", (step % 2) ? presentKeysValsAlt : presentKeysVals, firstLayerId);

        if (step == 0)
        {
            auto kvCacheShape = presentKeysValsAlt.at(0)->getShape();
            kvCacheShape.d[3] = 0;

            for (SizeType i = 0; i < localNbLayers; ++i)
            {
                std::string name = "past_key_value_" + std::to_string(firstLayerId + i);
                TensorPtr tmp = ITensor::view(presentKeysValsAlt.at(i), kvCacheShape);
                inputBuffers.insert_or_assign(name, std::move(tmp));
            }
        }
        else
        {
            utils::insertTensorVector(
                inputBuffers, "past_key_value_", (step % 2) ? presentKeysVals : presentKeysValsAlt, firstLayerId);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
