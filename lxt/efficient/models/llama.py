from functools import partial
from torch.nn import Dropout
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward

attnLRP = {
    LlamaMLP: partial(patch_method, gated_mlp_forward),
    LlamaRMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_llama: patch_attention,
}

cp_LRP = {
    LlamaMLP: partial(patch_method, cp_gated_mlp_forward),
    LlamaRMSNorm: partial(patch_method, rms_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_llama: patch_cp_attention,
}
