from functools import partial
from torch.nn import Dropout
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2RMSNorm

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward

attnLRP = {
    Qwen2MLP: partial(patch_method, gated_mlp_forward),
    Qwen2RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen2: patch_attention,
}

cp_LRP = {
    Qwen2MLP: partial(patch_method, cp_gated_mlp_forward),
    Qwen2RMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen2: patch_cp_attention,
}



