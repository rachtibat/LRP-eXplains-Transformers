from functools import partial
from torch.nn import Dropout, LayerNorm
from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import layer_norm_forward, dropout_forward
from lxt.efficient.rules import identity_rule_implicit


def mlp_forward(self, hidden_states):
    hidden_states = self.c_fc(hidden_states)
    hidden_states = identity_rule_implicit(self.act, hidden_states)
    hidden_states = self.c_proj(hidden_states)
    return hidden_states

attnLRP = {
    GPT2MLP: partial(patch_method, mlp_forward),
    modeling_gpt2.nn.LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_gpt2: patch_attention,
}

# CP-LRP is easier to use for GPT2, because GPT2 has negative logit values 
# and for AttnLRP we must also explain the softmax to kick out the negative bias
# so, we set cp_LRP as default for the average LXT users
cp_LRP = {
    GPT2MLP: partial(patch_method, mlp_forward),
    modeling_gpt2.nn.LayerNorm: partial(patch_method, layer_norm_forward),
    Dropout: partial(patch_method, dropout_forward),
    modeling_gpt2: patch_cp_attention, # only difference to attnLRP
}
