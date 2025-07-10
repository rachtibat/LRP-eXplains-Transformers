from functools import partial
from torch.nn import Dropout
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward

attnLRP = {
    Qwen3MLP: partial(patch_method, gated_mlp_forward),
    Qwen3RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen3: patch_attention,
}

cp_LRP = {
    Qwen3MLP: partial(patch_method, cp_gated_mlp_forward),
    Qwen3RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen3: patch_cp_attention,
}