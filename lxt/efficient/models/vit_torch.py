from functools import partial
from torchvision.models import vision_transformer

from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward, cp_multi_head_attention_forward

# AttnLRP outside the attention mechanism & CP-LRP inside the attention mechnism is easier to tune for gamma
cp_LRP = {
    vision_transformer.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    vision_transformer.nn.LayerNorm: partial(patch_method, layer_norm_forward),
    vision_transformer.nn.MultiheadAttention: partial(patch_method, cp_multi_head_attention_forward, keep_original=True),
}

