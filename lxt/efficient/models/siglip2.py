from functools import partial
import torch
from typing import Optional
from torch.nn import Dropout, LayerNorm
from transformers.models.siglip2 import modeling_siglip2
from transformers.models.siglip2.modeling_siglip2 import Siglip2MLP
from transformers.models.siglip2.modeling_siglip2 import (
    Siglip2MultiheadAttentionPoolingHead,
)
try:
    from transformers.models.siglip2.modeling_siglip2 import _prepare_4d_attention_mask
except ImportError:
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask


from lxt.efficient.patches import (
    identity_rule_implicit,
    patch_method,
    patch_cp_attention,
    patch_attention,
)
from lxt.efficient.patches import (
    layer_norm_forward,
    dropout_forward,
)
from lxt.efficient.rules import divide_gradient, stop_gradient


def siglip2_mlp_forward(self, hidden_state):
    hidden_state = self.fc1(hidden_state)
    hidden_state = identity_rule_implicit(self.activation_fn, hidden_state)
    hidden_state = self.fc2(hidden_state)
    return hidden_state


def siglip2_multihead_attention_pooling_head_forward(
    self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size = hidden_state.shape[0]
    probe = self.probe.repeat(batch_size, 1, 1)

    if attention_mask is not None:
        target_len, source_len = probe.shape[1], hidden_state.shape[1]
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, hidden_state.dtype, target_len
        )
        attention_mask = attention_mask.repeat(1, self.num_heads, target_len, 1)
        attention_mask = attention_mask.reshape(-1, target_len, source_len)

    hidden_state = self.attention(
        stop_gradient(probe),
        stop_gradient(hidden_state),
        hidden_state,
        attn_mask=attention_mask,
    )[0]

    hidden_state = divide_gradient(hidden_state, 2)
    residual = hidden_state
    hidden_state = self.layernorm(hidden_state)
    hidden_state = residual + self.mlp(hidden_state)

    return hidden_state[:, 0]


cp_LRP = {
    Siglip2MLP: partial(patch_method, siglip2_mlp_forward),
    Siglip2MultiheadAttentionPoolingHead: partial(
        patch_method, siglip2_multihead_attention_pooling_head_forward
    ),
    LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_siglip2: patch_cp_attention,
}

attnLRP = {
    Siglip2MLP: partial(patch_method, siglip2_mlp_forward),
    Siglip2MultiheadAttentionPoolingHead: partial(
        patch_method, siglip2_multihead_attention_pooling_head_forward
    ),
    LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_siglip2: patch_attention,
}
