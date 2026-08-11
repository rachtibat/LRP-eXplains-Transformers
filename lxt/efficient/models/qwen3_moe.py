import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.nn import Dropout
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP, Qwen3MoeRMSNorm, Qwen3MoeExperts

from lxt.efficient.patches import patch_method, patch_attention, patch_cp_attention
from lxt.efficient.patches import rms_norm_forward, gated_mlp_forward, cp_gated_mlp_forward, dropout_forward
from lxt.efficient.rules import divide_gradient, identity_rule_implicit, stop_gradient


def experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)

        current_hidden_states = identity_rule_implicit(self.act_fn, gate) * up ### <---- LXT
        current_hidden_states = divide_gradient(current_hidden_states, 2) ### <---- LXT

        current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

        current_hidden_states = divide_gradient(current_hidden_states, 2) ### <---- LXT

        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states



attnLRP = {
    Qwen3MoeMLP: partial(patch_method, gated_mlp_forward),
    Qwen3MoeExperts : partial(patch_method, experts_forward),
    Qwen3MoeRMSNorm: partial(patch_method, rms_norm_forward), 
    Dropout: partial(patch_method, dropout_forward),
    modeling_qwen3_moe: patch_attention,
}

# cp_LRP = {
#     Qwen3MoeMLP: partial(patch_method, cp_gated_mlp_forward),
#     Qwen3MoeRMSNorm: partial(patch_method, rms_norm_forward),
#     Dropout: partial(patch_method, dropout_forward),
#     modeling_qwen3_moe: patch_cp_attention,
# }