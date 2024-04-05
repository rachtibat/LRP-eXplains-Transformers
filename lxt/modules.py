import torch
import torch.nn as nn
import inspect

import lxt.functional as lf

###################
### LRP Modules ###
###################

class SoftmaxDT(nn.Softmax):

    def __init__(self, dim: int, inplace=False, **kwargs):
        super().__init__(dim)
        self.inplace = inplace

    def forward(self, inputs):
        return lf.softmax(inputs, self.dim, None, self.inplace)


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
        return lf.rms_norm_identity_fn.apply(hidden_states, self.weight, self.variance_epsilon)
    


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


def initialize_linear(original, replacement):
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


INIT_MODULE_MAPPING = {
    SoftmaxDT: initialize_generic,
    LinearEpsilon: initialize_linear,
    RMSNormIdentity: initialize_generic
}
