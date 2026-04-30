# Patched to skip bert (transformers.pytorch_utils API drift removed
# find_pruneable_heads_and_indices) and to register qwen3_5.
import warnings

import lxt.efficient.models.llama as llama
import lxt.efficient.models.qwen2 as qwen2
import lxt.efficient.models.qwen3 as qwen3
import lxt.efficient.models.qwen3_5 as qwen3_5
import lxt.efficient.models.gemma3 as gemma3
import lxt.efficient.models.gpt2 as gpt2
try:
    import lxt.efficient.models.vit_torch as vit_torch
except ModuleNotFoundError:
    vit_torch = None  # torchvision not installed; skipping (LLMs do not need vit_torch)

# bert intentionally skipped — the upstream lxt.efficient.models.bert imports
# `find_pruneable_heads_and_indices` from transformers.pytorch_utils, which
# was removed in transformers >=4.55. Re-enable when bert.py is fixed upstream
# or when transformers is pinned to a compatible version.
try:
    import lxt.efficient.models.bert as bert
    _BERT_OK = True
except Exception as e:
    warnings.warn(f"lxt.efficient.models.bert disabled: {e}")
    bert = None
    _BERT_OK = False


DEFAULT_MAP = {
    llama.modeling_llama: llama.attnLRP,
    qwen2.modeling_qwen2: qwen2.attnLRP,
    qwen3.modeling_qwen3: qwen3.attnLRP,
    qwen3_5.modeling_qwen3_5: qwen3_5.attnLRP,
    gemma3.modeling_gemma3: gemma3.attnLRP,
    gpt2.modeling_gpt2: gpt2.attnLRP,
}
if _BERT_OK:
    DEFAULT_MAP[bert.modeling_bert] = bert.attnLRP
if vit_torch is not None:
    DEFAULT_MAP[vit_torch.vision_transformer] = vit_torch.cp_LRP


def get_default_map(module):
    if module in DEFAULT_MAP:
        return DEFAULT_MAP[module]
    else:
        supported_models = ", ".join([key.__name__ for key in DEFAULT_MAP.keys()])
        raise ValueError(
            f"{module.__name__} not yet supported. Supported models are: {supported_models} "
            f"Please provide a custom patch_map. Contributions to the GitHub repository are welcome!"
        )
