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
from torch.autograd import Function


def identity_rule_implicit(fn, input):
    """
    Implements the identity rule (from Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on element-wise non-linear functions.

    Parameters:
    -----------

    fn: callable
        The function to be called with the input.
        This function must accept a single tensor as input and return a tensor of the same shape.
    input: torch.Tensor
        The input tensor
    """
    
    return identity_rule_implicit_fn.apply(fn, input)


def divide_gradient(input, factor=2):
    """
    Implements the uniform rule (from Equation 7 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on a tensor after the torch.matmul or the element-wise multiplication operation.

    Parameters:
    -----------
    input: torch.Tensor
        An input tensor
    factor: int
        The factor to divide the gradient/relevance by
    """

    return divide_gradient_fn.apply(input, factor)


def stop_gradient(input):
    """
    Stop the gradient from flowing through the input tensor.
    This rule is used in CP-LRP (from the paper 
    XAI for Transformers: Better Explanations through Conservative Propagation).
    """

    return input.detach()


class identity_rule_implicit_fn(Function):
    """
    Implements the identity rule (from Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on element-wise non-linear functions.

    Parameters:
    -----------

    fn: callable
        The function to be called with the input.
        This function must accept a single tensor as input and return a tensor of the same shape.
    input: torch.Tensor
        The input tensor
    """

    @staticmethod
    def forward(ctx, fn, input, epsilon=1e-10):

        output = fn(input)
        if input.requires_grad:
            ctx.save_for_backward(output/(input + epsilon))
        return output

    @staticmethod
    def backward(ctx, *out_relevance):

        gradient = ctx.saved_tensors[0] * out_relevance[0]

        return None, gradient, None


class divide_gradient_fn(Function):
    """
    Implements the uniform rule (from Equation 7 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers)
    in a more efficient manner through the Gradient*Input framework.

    Used on a tensor after the torch.matmul or the element-wise multiplication operation.

    Parameters:
    -----------
    input: torch.Tensor
        An input tensor
    factor: int
        The factor to divide the gradient/relevance by
    """

    @staticmethod
    def forward(ctx, input, factor=2):
        ctx.factor = factor
        return input

    @staticmethod
    def backward(ctx, *out_relevance):

        return out_relevance[0] / ctx.factor, None
    