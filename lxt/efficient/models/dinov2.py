from functools import partial
from torch.nn import Dropout, LayerNorm
from transformers.models.dinov2 import modeling_dinov2

from lxt.efficient.patches import (
    patch_method,
    patch_cp_attention,
    patch_attention,
)
from lxt.efficient.patches import (
    layer_norm_forward,
    dropout_forward,
)
#
#
# def dino_mlp_forward(self, hidden_state):
#     hidden_state = self.fc1(hidden_state)
#     hidden_state = identity_rule_implicit(self.activation, hidden_state)
#     hidden_state = self.fc2(hidden_state)
#     return hidden_state


attnLRP = {
    LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_dinov2: patch_attention,
}

cp_LRP = {
    LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_dinov2: patch_cp_attention,
}
