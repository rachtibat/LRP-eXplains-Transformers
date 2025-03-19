# Copyright 2024, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. &
# the authors: Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain,
# Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek. All rights reserved.
# 
# This code is based on the following work:
# 
#   'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers. ICML 2024.'
#
# The copyright in this software is being made available under the Clear BSD License.
# No patent rights, trademark rights and/or other Intellectual Property Rights other than
# the copyrights concerning the Software are granted under this license.
# You may obtain a full copy of the License at
#     
#   https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/LICENSE
#
import sys
import torch
from warnings import warn
from lxt.efficient.rules import stop_gradient, divide_gradient, identity_rule_implicit


def check_already_patched(target_fn, new_fn):
    """
    Check if a function is already replaced by another function.
    Used to avoid redundant patching.

    Parameters
    ----------
    target_fn : function
        The function to be wrapped.
    new_fn : function
        The new function to wrap the target function.

    Returns
    -------
    bool
        True if the target function is already wrapped by the new function, False otherwise.
    """

    if target_fn.__module__ == new_fn.__module__:
        if hasattr(target_fn, '__name__'):
            warn(f"{target_fn.__name__} already patched.")
        else:
            warn(f"{target_fn} already patched.")
        return True
    return False


def patch_method(fn, module, method_name="forward", keep_original=False):
    """
    Patch a method in a module with a new function.

    Parameters
    ----------
    fn : function
        The function to replace the method with.
    module : module
        The module containing the method to be patched.
    method_name : str, optional
        The name of the method to be patched. Default is "forward".
    keep_original : bool, optional
        If True, the original method is saved in the module as f"original_{method_name}".
        Default is False.
    
    Returns
    -------
    bool
        True if the method was successfully patched, False otherwise.
    """
    
    if check_already_patched(getattr(module, method_name), fn):
        return False
    if keep_original:
        setattr(module, f'original_{method_name}', getattr(module, method_name))
    
    setattr(module, method_name, fn)
    return True


def replace_module(patched_module, original_module):
    """
    Replace all attributes of a module with the attributes of another module.

    Parameters
    ----------
    patched_module : module
        The module whose attributes will be copied to the original module.
    original_module : module
        The module whose attributes will be replaced by the patched module.
    
    Returns
    -------
    bool
        True if the module was successfully patched, False otherwise.
    """
    if original_module == patched_module:
        return False

    # Then replace all attributes
    for attr in dir(patched_module):
        if not attr.startswith('__'):  # Skip special methods
            setattr(original_module, attr, getattr(patched_module, attr))

    return True


#############################
###### AttnLRP Patches ######
#############################

def rms_norm_forward(self, hidden_states):
    """
    On normalization operations, we apply the identity rule.
    It is implemented here by stopping the gradient flow through the variance calculation,
    which is equivalent to the identity rule in a Gradient*Input framework.
    """

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * stop_gradient(torch.rsqrt(variance + self.variance_epsilon))

    return self.weight * hidden_states.to(input_dtype)


def layer_norm_forward(self, x):
    """
    On normalization operations, we apply the identity rule.
    It is implemented here by stopping the gradient flow through the variance calculation,
    which is equivalent to the identity rule in a Gradient*Input framework.
    """

    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    std = (var + self.eps).sqrt()
    y = (x - mean) / stop_gradient(std)
    if self.weight is not None:
        y *= self.weight
    if self.bias is not None:
        y += self.bias

    return y


def gated_mlp_forward(self, x):
    """
    On the element-wise non-linear activation, we apply the identity rule and
    on the element-wise multiplication, we apply the uniform rule.
    Both rules are implemented via the Gradient*Input framework.
    """

    gate_out = self.gate_proj(x)
    gate_out = identity_rule_implicit(self.act_fn, gate_out)

    weighted = gate_out * self.up_proj(x)
    weighted = divide_gradient(weighted, 2)
    return self.down_proj(weighted)


def mlp_forward(self, x):
    """
    On the element-wise non-linear activation, we apply the identity rule,
    which is implemented via the Gradient*Input framework.
    """

    up_out = self.up_proj(x)
    up_out = identity_rule_implicit(self.act_fn, up_out)
    return self.down_proj(up_out)


def patch_attention(module):
    """
    Huggingface's transformers library provides a dictionary of all attention functions.
    We patch all of them with the same wrapper function to implement the uniform rule in
    matmul operations via the Gradient*Input framework. It is sufficient to correct the
    gradient flow later at the query, key, and value tensors.
    """
    new_forward = wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    else:
        module.eager_attention_forward = new_forward
    
    NEW_ATTENTION_FUNCTIONS = {}
    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        new_forward = wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        else:
            NEW_ATTENTION_FUNCTIONS[key] = new_forward
    module.ALL_ATTENTION_FUNCTIONS = NEW_ATTENTION_FUNCTIONS
            #module.ALL_ATTENTION_FUNCTIONS[key] = new_forward
    return True


def wrap_attention_forward(forward_fn):
    def attention_forward(module, query, key, value, *args, **kwargs):

        query = divide_gradient(query, 4)
        key = divide_gradient(key, 4)
        value = divide_gradient(value, 2)

        if 'dropout' in kwargs:
            kwargs['dropout'] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)
    return attention_forward


def non_linear_forward(self, x):
    """
    Patch the element-wise non-linear activation functions with the identity rule,
    which is implemented via the Gradient*Input framework.
    """
    return identity_rule_implicit(self.original_forward, x)


def dropout_forward(self, x):
    """
    To use gradient checkpointing in huggingface, we must set the model to 'train()' mode.
    However, this will also activate the dropout layers. We patch the dropout layers
    to set the dropout rate to zero during the forward pass.
    """
    return x



############################
###### CP-LRP Patches ######
############################

def patch_cp_attention(module):
    """
    For the CP-LRP variant, no gradient is allowed to flow through the softmax function.
    We patch all attention functions with a wrapper function that stops the gradient flow
    at the query and key tensors, which are directly connected to the softmax function.
    """
    new_forward = cp_wrap_attention_forward(module.eager_attention_forward)
    if check_already_patched(module.eager_attention_forward, new_forward):
        return False
    else:
        module.eager_attention_forward = new_forward
    
    for key, value in module.ALL_ATTENTION_FUNCTIONS.items():
        new_forward = cp_wrap_attention_forward(value)
        if check_already_patched(value, new_forward):
            return False
        else:
            module.ALL_ATTENTION_FUNCTIONS[key] = new_forward
    return True


def cp_wrap_attention_forward(forward_fn):
    def cp_attention_forward(module, query, key, value, *args, **kwargs):

        query = stop_gradient(query)
        key = stop_gradient(key)

        if 'dropout' in kwargs:
            kwargs['dropout'] = 0.0
        return forward_fn(module, query, key, value, *args, **kwargs)
    return cp_attention_forward


def cp_multi_head_attention_forward(self, query, key, value, *args, **kwargs):
    """
    For the CP-LRP variant, no gradient is allowed to flow through the softmax function.
    We patch the torch.nn.MultiheadAttention.forward such that the gradient flow
    at the query and key tensors is stopped, which are directly connected to the softmax function.
    """
    query = stop_gradient(query)
    key = stop_gradient(key)
    return self.original_forward(query, key, value, *args, **kwargs)


def cp_gated_mlp_forward(self, x):
    """
    For the CP-LRP variant, no gradient is allowed to flow through the gating mechanism.
    """
    gate_out = stop_gradient(self.gate_proj(x))
    gate_out = self.act_fn(gate_out)
    
    weighted = gate_out * self.up_proj(x)
    return self.down_proj(weighted)
