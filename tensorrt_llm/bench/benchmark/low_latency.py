from __future__ import annotations

import asyncio
import os
from copy import deepcopy
from pathlib import Path
from random import choices, shuffle
from time import monotonic_ns, sleep
from typing import List

import click
import yaml
from click_option_group import optgroup

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.benchmark.utils.asynchronous import async_benchmark
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.bench.dataclasses.enums import IFBSchedulingPolicy
from tensorrt_llm.bench.dataclasses.general import BenchmarkEnvironment
from tensorrt_llm.bench.dataclasses.reporting import (StatsKeeper,
                                                      report_latency_statistics)
from tensorrt_llm.bench.dataclasses.statistics import BenchmarkStatistics
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

# isort: off
from tensorrt_llm.bench.benchmark.utils.general import (get_executor_requests,
                                                        get_settings_from_engine
                                                        )
# isort: on
from tensorrt_llm.bench.utils.data import (create_dataset_from_stream,
                                           initialize_tokenizer)
from tensorrt_llm.logger import logger


@click.command(name="latency")
@optgroup.group("Engine run configuration",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--engine_dir",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    required=True,
    help="Path to a serialized TRT-LLM engine.",
)
@optgroup.option(
    "--kv_cache_free_gpu_mem_fraction",
    type=float,
    default=.90,
    help="The percentage of memory to use for KV Cache after model load.",
)
@optgroup.group(
    "Engine Input Configuration",
    help="Input configuration for driving the engine.",
)
@optgroup.option(
    "--dataset",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    help="Pass in a dataset file for parsing instead of stdin.",
)
@optgroup.option(
    "--num_requests",
    type=int,
    default=0,
    help="Number of requests to cap benchmark run at. Minimum between value and"
    "length of dataset.",
)
@optgroup.option(
    "--warmup",
    type=int,
    default=0,
    help="Number of requests warm up benchmark.",
)
@optgroup.group("Speculative Decode Options",
                help="Runtime settings for executing a TensorRT-LLM engine.")
@optgroup.option(
    "--medusa_choices",
    type=click.Path(exists=True,
                    readable=True,
                    path_type=Path,
                    resolve_path=True),
    default=None,
    required=False,
    help="Path to a YAML file that defines the Medusa tree.",
)
@click.pass_obj
def latency_command(
    bench_env: BenchmarkEnvironment,
    **params,
) -> None:
    """Run a latency test on a TRT-LLM engine."""

    logger.set_level("info")
    logger.info("Preparing to run latency benchmark...")
    # Parameters from CLI
    # Model, experiment, and engine params
    dataset_path: Path = params.pop("dataset")
    num_requests: int = params.pop("num_requests")
    model: str = bench_env.model
    checkpoint_path: Path = bench_env.model_path or bench_env.model
    engine_dir: Path = params.pop("engine_dir")
    # Engine configuration parsing
    exec_settings, build_cfg = get_settings_from_engine(engine_dir)
    exec_settings["model"] = model
    engine_tokens = exec_settings["settings_config"]["max_num_tokens"]
    engine_max_seq_len = build_cfg["max_seq_len"]

    # Runtime Options
    kv_cache_percent = params.pop("kv_cache_free_gpu_mem_fraction")
    medusa_choices = params.pop("medusa_choices")

    # Update configuration with runtime options
    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = 1
    exec_settings["settings_config"]["max_num_tokens"] = engine_tokens
    exec_settings["settings_config"]["beam_width"] = 1
    exec_settings["settings_config"]["chunking"] = False
    exec_settings["settings_config"][
        "scheduler_policy"] = IFBSchedulingPolicy.NO_EVICT

    # Set environment variables for setting runtime options.
    # TODO: Once passing of variables is fixed, these should work
    # when using MPI in C++ runtime.
    os.environ["TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG"] = "1"
    os.environ["TRTLLM_MMHA_KERNEL_BLOCK_SIZE"] = "256"
    os.environ["FORCE_MULTI_BLOCK_MODE"] = "1"
    os.environ["TRTLLM_ENABLE_PDL"] = "1"

    # Performance options
    exec_settings["performance_options"]["cuda_graphs"] = True
    exec_settings["performance_options"]["multi_block_mode"] = True

    # Decoding Options
    if medusa_choices is not None:
        with open(medusa_choices, "r") as medusa_yml:
            exec_settings["decoding_config"]["medusa_choices"] = \
                yaml.load(medusa_yml, Loader=yaml.SafeLoader)

    # Construct the runtime configuration dataclass.
    runtime_config = RuntimeConfig(**exec_settings)
    warmup_steps = params.get("warmup")

    # Initialize the HF tokenizer for the specified model.
    ignore_eos = True if runtime_config.decoding_config.decoding_mode == SpeculativeDecodingMode.NONE else False
    tokenizer = initialize_tokenizer(checkpoint_path)
    eos_id = tokenizer.eos_token_id if not ignore_eos else -1
    pad_id = tokenizer.pad_token_id if not ignore_eos else -1

    # Dataset Loading and Preparation
    with open(dataset_path, "r") as dataset:
        metadata, requests = create_dataset_from_stream(
            tokenizer, dataset, num_requests=num_requests)

    if metadata.max_sequence_length > engine_max_seq_len:
        raise RuntimeError(
            f"Engine supports a max sequence of {engine_max_seq_len}. Provided "
            "dataset contains a maximum sequence of "
            f"{metadata.max_sequence_length}. Please rebuild a new engine to"
            "support this dataset.")

    if not os.environ.get("TRTLLM_BENCH_EXPERIMENT", False):

        # Dataset Loading and Preparation
        executor_requests = get_executor_requests(
            requests,
            True,
            eos_id=eos_id,
            pad_id=pad_id,
        )

        # Instantiate the low latency benchmark.
        benchmark = LatencyBenchmark(
            executor_requests,
            runtime_config,
        )

        try:
            logger.info("Ready to start benchmark.")
            benchmark.setup_warmup(warmup_steps)
            benchmark.start_benchmark()
            benchmark.report_statistics()
        except KeyboardInterrupt:
            logger.info("Benchmark interrupted! Shutting down...")
        finally:
            benchmark.stop_benchmark()

    else:
        logger.info("Running experimental latency benchmark.")

        run_latency(requests, runtime_config)


def run_latency(requests, runtime_config: RuntimeConfig):
    # check setup
    assert runtime_config.settings_config.max_batch_size == 1
    assert runtime_config.settings_config.chunking is False

    asyncio.run(
        async_benchmark(runtime_config,
                        requests,
                        streaming=True,
                        concurrency=1,
                        for_latency=True))


class LatencyBenchmark:
    """Latency benchmark utility class."""

    def __init__(
        self,
        dataset: List[trtllm.Request],
        runtime_cfg: RuntimeConfig,
    ) -> None:
        """Initialize the throughput benchmark.

        Args:
            dataset (List[trtllm.Request]): A dataset of TRT-LLM requests to
            benchmark against.
            runtime_cfg (RuntimeConfig): Runtime configuration.
        """
        # Dataset and input properties.
        self.requests = dataset
        self.warm_up_dataset = None
        self.runtime_config: RuntimeConfig = deepcopy(runtime_cfg)
        self.streaming = True

        # Benchmark stats and time tracking.
        self.start_time = None
        self.end_time = None
        self.submitted_requests = 0
        self.statistics = StatsKeeper()

        logger.info("Starting Executor backend...")
        self.executor = None
        logger.info("Executor started.")

    def setup_warmup(self, steps) -> None:
        """Warm up the benchmarker."""
        if steps > 0:
            self.warm_up_dataset = choices(self.requests, k=steps)
            shuffle(self.warm_up_dataset)

    def start_benchmark(self) -> None:
        """Start the benchmark."""
        logger.info("Initializing backend...")
        self.executor = trtllm.Executor(
            self.runtime_config.engine_dir,
            trtllm.ModelType.DECODER_ONLY,
            executor_config=self.runtime_config.get_config())

        logger.info("WAITING ON EXECUTOR...")
        while not self.executor.can_enqueue_requests():
            logger.info("Waiting for executor to stand up...")
            sleep(1)

        if self.warm_up_dataset and len(self.warm_up_dataset) > 0:
            logger.info(f"WARMING UP...")
            for i, request in enumerate(self.warm_up_dataset, start=1):
                logger.info(f"Running warm up step {i}...")
                req_id = self.executor.enqueue_request(request)
                final = False
                while not final:
                    responses = self.executor.await_responses(req_id)
                    final = any([resp.result.is_final for resp in responses])

            logger.info("WARMUP COMPLETE.")

        logger.info("Low latency benchmark started.")
        self.start_time = monotonic_ns()
        while len(self.requests) > 0:
            final = False
            request = self.requests.pop(0)

            req_id = self.executor.enqueue_request(request)
            self.statistics.register_request(req_id, monotonic_ns(),
                                             len(request.input_token_ids))

            while not final:
                responses = self.executor.await_responses(req_id)
                now = monotonic_ns()
                for resp in responses:
                    self.statistics.register_response(
                        req_id,
                        now,
                        resp.result.is_final,
                        resp.has_error(),
                        resp.result.decoding_iter,
                        resp.result.output_token_ids[0],
                        time_on_first_token=now - self.start_time)
                    final = resp.result.is_final

        self.end_time = monotonic_ns()
        logger.info("Low latency benchmark finished.")

    def stop_benchmark(self) -> None:
        """Stop the benchmark and clean up backend and threads."""
        logger.info("Benchmark Shutdown called!")
        if self.executor is not None:
            self.executor.shutdown()
        logger.info("Executor shutdown.")

    def report_statistics(self) -> BenchmarkStatistics:
        """Report internal statistics about benchmark."""

        report_latency_statistics(self.statistics, self.runtime_config, logger)

        return self.statistics
