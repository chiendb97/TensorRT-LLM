from contextlib import contextmanager
from typing import Dict, List, Tuple

# isort: off
import torch
# isort: on

import datasets
from datasets import load_dataset


def prepare_image_inputs(batch_size=8):
    pass


def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)


def intermediate_check(tllm_inter: Dict, hf_ref: Tuple[torch.Tensor], attn_mask,
                       logger):

    def apply_mask(x):
        return x * attn_mask

    # minus one because there is an embedding output
    num_layers = len(hf_ref) - 1

    res = tllm_inter['embedding_output']
    res = apply_mask(res)
    ref = hf_ref[0]
    ref = apply_mask(ref)
    torch.testing.assert_close(actual=res, expected=ref, rtol=1e-2, atol=1e-2)
    logger.debug("Embedding are all close")

    for i in range(num_layers - 1):
        res = tllm_inter[f'layer_{i}_output']
        res = apply_mask(res)
        ref = hf_ref[i + 1]
        ref = apply_mask(ref)
        is_close = torch.allclose(res, ref, rtol=1e-2, atol=1e-2)
        logger.debug(f'BertEncoderLayer_{i}_output is close: {is_close}')


@contextmanager
def temporary_datasets_config(**kwargs):
    # Save original settings
    original_settings = {}
    for key, value in kwargs.items():
        original_settings[key] = getattr(datasets.config, key)
        setattr(datasets.config, key, value)
    try:
        yield
    finally:
        # Restore original settings
        for key, value in original_settings.items():
            setattr(datasets.config, key, value)
