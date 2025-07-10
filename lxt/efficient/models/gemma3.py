import torch
from functools import partial
from torch.nn import Dropout
from transformers.models.gemma3 import modeling_gemma3
from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP, Gemma3RMSNorm

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import gated_mlp_forward, cp_gated_mlp_forward, dropout_forward
from lxt.efficient.rules import stop_gradient

def gemma3_norm(self, x):
    return x * stop_gradient(torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

attnLRP = {
    Gemma3MLP: partial(patch_method, gated_mlp_forward),
    Gemma3RMSNorm: partial(patch_method, gemma3_norm, method_name="_norm"),
    Dropout: partial(patch_method, dropout_forward),
    modeling_gemma3: patch_attention,
}

cp_LRP = {
    Gemma3MLP: partial(patch_method, cp_gated_mlp_forward),
    Gemma3RMSNorm: partial(patch_method, gemma3_norm, method_name="_norm"),
    Dropout: partial(patch_method, dropout_forward),
    modeling_gemma3: patch_cp_attention,
}