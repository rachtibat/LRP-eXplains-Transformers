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
import torch
from warnings import warn

try:
    from zennit.core import ParamMod, stabilize, BasicHook
except ImportError:
    warn("'zennit' library is not available. Please install it to use for vision transformers.")
from lxt.efficient.patches import patch_method


def forward(self, module, input, output):
    '''Forward hook to save module in-/outputs.'''
    self.stored_tensors['input'] = input
    self.stored_tensors['output'] = output # <--- NEW


def backward(self, module, grad_input, grad_output):
    '''Backward hook to compute LRP based on the class attributes.'''

    # multiply relevance by output to convert to standard LRP relevance scores
    assert len(grad_output) == 1, 'Only single output supported for now!' # <--- NEW
    grad_output = (grad_output[0] * self.stored_tensors['output'],) # <--- NEW
    grad_output[0].requires_grad = True # <--- NEW

    original_input = self.stored_tensors['input'][0].clone()
    inputs = []
    outputs = []
    for in_mod, param_mod, out_mod in zip(self.input_modifiers, self.param_modifiers, self.output_modifiers):
        input = in_mod(original_input).requires_grad_()
        with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
            output = modified.forward(input)
            output = out_mod(output)
        inputs.append(input)
        outputs.append(output)
    grad_outputs = self.gradient_mapper(grad_output[0], outputs)
    gradients = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        create_graph=grad_output[0].requires_grad
    )
    relevance = self.reducer(inputs, gradients)

    # divide relevance by input to revert to gradient*input framework
    relevance = relevance / stabilize(original_input, epsilon=1e-10) # <--- NEW

    return tuple(relevance if original.shape == relevance.shape else None for original in grad_input)


def monkey_patch_zennit(verbose=False):
    """
    This module modifies the zennit library to support the gradient*input framework for Layer-Wise Relevance Propagation (LRP) in transformers.
    1. It modifies the 'forward hook' in zennit to additionally save module outputs for later use in backward pass.
    2. It modifies the 'backward hook' in zennit to support the gradient*input framework for LRP by multiplying the gradient output with the module output
        and dividing the output relevance by the input.
    """
    for method, name in [(forward, "forward"), (backward, "backward")]:
        success = patch_method(method, BasicHook, method_name=name)
        if not success:
            warn(f"Failed to patch Zennit BasicHook's {name}. Skipping...")
        elif verbose:
            print(f"Patched Zennit BasicHook's {name}")

