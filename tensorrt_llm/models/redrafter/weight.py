
import recurrent_drafting # type: ignore
import torch
from transformers import AutoConfig, LlamaForCausalLM, AutoModel


def load_llm(hf_llm: str, dtype: torch.dtype, device: torch.device):
    cfg = AutoConfig.from_pretrained(hf_llm)
    assert cfg.model_type == "llama"
    return LlamaForCausalLM.from_pretrained(
        hf_llm, torch_dtype=dtype
    ).to(device)


def load_drafter(
    hf_drafter_dir: str, dtype: torch.dtype, device: torch.device):
    recurrent_drafting.modeling_drafter.register_auto_models()
    return AutoModel.from_pretrained(hf_drafter_dir, torch_dtype=dtype).to(device)


# binhtt4: deprecated.
# used to load redrafter and run simple forward for quantizing.
def load_redrafter_hf(model_dir, drafter_model_dir, dtype, device):
    print("loading redrafting model...")
    llm = load_llm(model_dir, dtype, device)
    drafter = load_drafter(drafter_model_dir, dtype, device)
    model = recurrent_drafting.recurrent_drafting.RecurrentDrafting(llm=llm, drafter=drafter)
    return model
