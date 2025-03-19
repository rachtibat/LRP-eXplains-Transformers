import torch
import torch.nn as nn
import inspect
import lxt.explicit.functional as lf
import lxt.explicit.special as ls
import torch.fx


###################
### LRP Modules ###
###################

class SoftmaxDT(nn.Softmax):

    def __init__(self, dim: int, dtype=None, temperature=1.0, inplace=False, **kwargs):
        super().__init__(dim)
        self.inplace = inplace
        self.dtype = dtype
        self.temperature = temperature

    def forward(self, inputs):
        return lf.softmax(inputs, self.dim, self.dtype, self.temperature, self.inplace)


class LinearEpsilon(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, epsilon=1e-6, **kwargs):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.epsilon = epsilon

    def forward(self, inputs):
        return lf.linear_epsilon(inputs, self.weight, self.bias, self.epsilon)
    

class RMSNormIdentity(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return lf.rms_norm_identity(hidden_states, self.weight, self.variance_epsilon)
    

class LayerNormEpsilon(nn.LayerNorm):

    def __init__(self, normalized_shape, eps: float = 0.00001, elementwise_affine: bool = True, bias: bool = True, device=None, dtype=None):
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)

    def forward(self, x):
        return lf.layer_norm(x, self.weight, self.bias, self.eps)
    

##################################
### MultiheadAttention Modules ###
##################################

class LinearInProjection(nn.Module):
    """
    Custom nn.Linear module to make it easier to attach different rules to it.
    """
    def __init__(self, weight, bias):
        super().__init__()

        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)
    
class LinearOutProjection(nn.Module):
    """
    Custom nn.Linear module to make it easier to attach different rules to it.
    """
    def __init__(self, weight, bias):
        super().__init__()

        self.weight = weight
        self.bias = bias
    
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)

class MultiheadAttention_CP(nn.Module):
    """
    Implementing the CP-LRP (Conservative Propagation - LRP) rule for attention i.e. we don't let relevance flow through the softmax, but only through the value path. 
    This method *only works well in Vision Transformers* because here the advanced AttnLRP rules, which do use the softmax, have similar performance to CP-LRP rules. 
    The issue with AttnLRP is that using the softmax introduces gradient shattering, which requires applying the z-plus LRP rule. 
    This makes AttnLRP slightly less efficient and, based on our limited experiments, the small performance gain is not worthwhile in Vision Transformers.
    However, in Large Language Models, applying AttnLRP on the softmax is substantially better than CP-LRP and does not require the less efficient z-plus rule.
    Therefore, we choose the more efficient CP-LRP for attention and use AttnLRP for other parts of the ViT.

    Please refer to Section A.2.3. 'Tackling Noise in Vision Transformers' of the the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'.
    """
    def __init__(self):
        super().__init__()

        self.q_proj_weight = None
        self.k_proj_weight = None
        self.v_proj = LinearInProjection(None, None)
        self.out_proj = LinearOutProjection(None, None)

        self.embed_dim = None
        self.num_heads = None
        self.head_dim = None
        self.batch_first = None

        self.bias_q = None
        self.bias_k = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True, is_causal=False):

        assert is_causal == False # not supported yet

        return ls.multi_head_attention_cp(query, key, value, self.batch_first, self.num_heads, self.head_dim, self.q_proj_weight, self.bias_q, self.k_proj_weight, 
                                self.bias_k, self.v_proj, self.out_proj, key_padding_mask, need_weights, attn_mask, average_attn_weights)
    
    
##########################
### Initialize Modules ###
##########################

def copy_parameters_and_buffers_(original, replacement):
    """
    Copy the parameters and buffers of the original module.
    """

    for name, param in original.named_parameters():
        replacement.register_parameter(name, param)

    for name, buffer in original.named_buffers():
        replacement.register_buffer(name, buffer)


def initialize_generic(original, replacement):
    """
    Initialize a replacement module with the correct arguments.
    """
    
    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    replacement = replacement(**kwargs)
    copy_parameters_and_buffers_(original, replacement)

    return replacement


def initialize_bias(original, replacement):
    """
    Initialize the LinearEpsilon module correctly with the bias argument.
    """

    kwargs = {}
    for arg in inspect.signature(original.__init__).parameters.keys():
        if hasattr(original, arg):
            kwargs[arg] = getattr(original, arg)

    kwargs["bias"] = True if original.bias is not None else False

    replacement = replacement(**kwargs)
    copy_parameters_and_buffers_(original, replacement)

    return replacement


def initialize_MHA(original, replacement):
    """
    Initialize a MultiheadAttention_CP module.
    """
    
    replacement = replacement()
    
    if not original._qkv_same_embed_dim:
        replacement.q_proj_weight = original.q_proj_weight
        replacement.k_proj_weight = original.k_proj_weight
        replacement.v_proj.weight = original.v_proj.weight
    else:
        replacement.q_proj_weight = original.in_proj_weight[:original.embed_dim]
        replacement.k_proj_weight = original.in_proj_weight[original.embed_dim:original.embed_dim*2]
        replacement.v_proj.weight = original.in_proj_weight[original.embed_dim*2:original.embed_dim*3]

    if original.in_proj_bias is not None:
        replacement.bias_q = original.in_proj_bias[:original.embed_dim]
        replacement.bias_k = original.in_proj_bias[original.embed_dim:original.embed_dim*2]
        replacement.v_proj.bias = original.in_proj_bias[original.embed_dim*2:original.embed_dim*3]
        
    if original.bias_k is not None:
        raise NotImplementedError("add_bias_kv=True is not supported yet.")
    
    replacement.out_proj.weight = original.out_proj.weight
    replacement.out_proj.bias = original.out_proj.bias

    replacement.embed_dim = original.embed_dim
    replacement.num_heads = original.num_heads
    replacement.head_dim = original.head_dim
    replacement.batch_first = original.batch_first

    return replacement


INIT_MODULE_MAPPING = {
    SoftmaxDT: initialize_generic,
    LinearEpsilon: initialize_bias,
    RMSNormIdentity: initialize_generic,
    LayerNormEpsilon: initialize_bias,
    MultiheadAttention_CP: initialize_MHA,
}
