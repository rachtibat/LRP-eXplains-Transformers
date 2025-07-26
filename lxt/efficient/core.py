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
from warnings import warn
from lxt.efficient.models import get_default_map


def monkey_patch(module, patch_map=None, verbose=False):
    """
    This function modifies the module's classes with the provided patch_map by e.g. replacing the forward method.
    This way, Layer-wise Relevance Propagation rules can be applied to the module's layers.

    Parameters:
    -----------
    module: Python module
        The module to be patched.
    patch_map: dict
        A dictionary that maps the target classes to the patch functions. If None, a default patch_map is used.
        The patch functions should take the target class as input and return True if the patching was successful.
    verbose: bool
        If True, prints the patched classes.
    """
    if patch_map is None:
        patch_map = get_default_map(module)

    for target, patch in patch_map.items():
        success = patch(target)
        if not success:
            warn(f"Failed to patch {target.__name__}. Skipping...")
        elif verbose:
            print(f"Patched {target.__name__}")


def zero_bias(model):
    """
    This function modifies your model to verify that

    Parameters:
    ----------
    model: Pytorch model
        The monkey_patch/modified model to zero the biases for. Key assumptions: all linear layers are initiated with torch.nn.Linear,
        all LayerNorms are initialized with torch.nn.LayerNorm, and all softmaxs are initiated with torch.nn.Softmax.
        A key point to note is that the model must already have modified the forward pass for all nonlinear activations on the inputs
        and for any bilinear matrix multiplications (any matrix multiplication that has components that can be traced back to the input
        embeddings. E.g. H*W_q x H*W_k in multiplying queries and keys in the standard attention mechanism (SDPA)).
        To verify that conservation truly holds for your model, confirm that the output logit in the model is approximately the same as
        the input summed across the embedding dimension.
    """
    import torch.nn as nn
    from torch.nn import Linear, LayerNorm, Softmax

    class DetachedSoftmax(nn.Module):
        def __init__(self, original_softmax):
            super().__init__()
            self.softmax = original_softmax

        def forward(self, x):
            return self.softmax(x).detach()

    for name, module in model.named_modules():
        if isinstance(module, Linear) or isinstance(module, LayerNorm):
            module.bias = None

        if isinstance(module, Softmax):
            parent = model
            attrs = name.split(".")
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)

            setattr(parent, attrs[-1], DetachedSoftmax(module))
