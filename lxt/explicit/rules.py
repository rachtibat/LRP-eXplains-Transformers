import torch
from torch.autograd import Function
import torch.nn as nn
from lxt.explicit.functional import _stabilize, conservation_check_wrap
from torch.func import jvp, vjp
import torch.fx

class WrapModule(nn.Module):
    """
    Base class for wrapping a rule around a module. This class is not meant to be used directly, but to be subclassed by specific rules.
    It is then used to replace the original module with the rule-wrapped module.
    """

    def __init__(self, module):
        super(WrapModule, self).__init__()
        self.module = module


class IdentityRule(WrapModule):
    """
    Distribute the relevance 100% to the input according to the identity rule in Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    Parameters:
    -----------

    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch and has a single input and output
    input: torch.Tensor
        The input tensor
    """

    def forward(self, input):

        return identity_fn.apply(self.module, input)
    

def identity(fn, input):
    """
    Distribute the relevance 100% to the input according to the identity rule in Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    Parameters:
    -----------

    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch and has a single input and output
    input: torch.Tensor
        The input tensor
    """
    return identity_fn.apply(fn, input)


class identity_fn(Function):
    """
    Distribute the relevance 100% to the input according to the identity rule in Equation 9 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    Parameters:
    -----------

    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch and has a single input and output
    input: torch.Tensor
        The input tensor
    """

    @staticmethod
    def forward(ctx, fn, input):

        output = fn(input)
        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return (None,) + out_relevance
    

class StopRelevanceRule(WrapModule):
    """
    Stop the relevance flow at the input. Equivalent to .detach() in PyTorch.

    Parameters:
    -----------

    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch and has a single input and output
    input: torch.Tensor
        The input tensor
    """

    def forward(self, input):

        return stop_relevance_fn.apply(self.module, input)
    

class stop_relevance_fn(Function):
    """
    Stop the relevance flow at the input. Equivalent to .detach() in PyTorch.

    Parameters:
    -----------

    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch and has a single input and output
    input: torch.Tensor
        The input tensor
    """

    @staticmethod
    def forward(ctx, fn, input):

        output = fn(input)
        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return (None, None)


class EpsilonRule(WrapModule):
    """
    Gradient X Input (Taylor Decomposition with bias or standard Epsilon-LRP rule for linear layers) according to the Equation 4-5 and 8 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    module: nn.Module
        The module to be wrapped
    epsilon: float
        Small value to stabilize the denominator in the input_x_gradient rule

    """

    def __init__(self, module, epsilon=1e-8):
        
        super(EpsilonRule, self).__init__(module)
        self.epsilon = epsilon

    def forward(self, *inputs):

        return epsilon_lrp_fn.apply(self.module, self.epsilon, *inputs)

@torch.fx.wrap
def epsilon_lrp(fn, epsilon, *inputs):
    """
    Gradient X Input (Taylor Decomposition with bias or standard Epsilon-LRP rule for linear layers) according to the Equation 4-5 and 8 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch
    epsilon: float
        Small value to stabilize the denominator
    *inputs: at least one torch.Tensor
        The input tensors to the function
    """
    return epsilon_lrp_fn.apply(fn, epsilon, *inputs)


class epsilon_lrp_fn(Function):
    """
    Gradient X Input (Taylor Decomposition with bias or standard Epsilon-LRP rule for linear layers) according to the Equation 4-5 and 8 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch
    epsilon: float
        Small value to stabilize the denominator
    *inputs: at least one torch.Tensor
        The input tensors to the function
    """

    @staticmethod
    def forward(ctx, fn, epsilon, *inputs):

        # create boolean mask for inputs requiring gradients
        #TODO: use ctx.needs_input_grad instead of requires_grad
        requires_grads = [True if inp.requires_grad else False for inp in inputs]
        if sum(requires_grads) == 0:
            # no gradients to compute or gradient checkpointing is used
            return fn(*inputs)
        
        # detach inputs to avoid overwriting gradients if same input is used as multiple arguments (like in self-attention)
        inputs = tuple(inp.detach().requires_grad_() if inp.requires_grad else inp for inp in inputs)

        with torch.enable_grad():
            outputs = fn(*inputs)

        ctx.epsilon, ctx.requires_grads = epsilon, requires_grads
        # save only inputs requiring gradients
        inputs = tuple(inputs[i] for i in range(len(inputs)) if requires_grads[i])
        ctx.save_for_backward(*inputs, outputs)
        
        return outputs.detach()
        
    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, outputs = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        relevance_norm = out_relevance[0] / _stabilize(outputs, ctx.epsilon, inplace=False)

        # computes vector-jacobian product
        grads = torch.autograd.grad(outputs, inputs, relevance_norm)

        # return relevance at requires_grad indices else None
        relevance = iter([grads[i].mul_(inputs[i]) for i in range(len(inputs))])
        return (None, None) + tuple(next(relevance) if req_grad else None for req_grad in ctx.requires_grads)

        
    

class UniformEpsilonRule(WrapModule):
    """
    A sequential application of the input_x_gradient rule and the uniform rule to distribute the relevance uniformly to all inputs according to the uniform rule 
    as discussed in Section 3.3.2. 'Handling Matrix-Multiplication' of the paper 'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'.

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    module: nn.Module
        The module to be wrapped
    epsilon: float
        Small value to stabilize the denominator in the input_x_gradient rule

    """

    def __init__(self, module, epsilon=1e-6):
        
        super(UniformEpsilonRule, self).__init__(module)
        self.epsilon = epsilon

    def forward(self, *inputs):

        return uniform_epsilon_lrp_fn.apply(self.module, self.epsilon, *inputs)
    

class uniform_epsilon_lrp_fn(epsilon_lrp_fn):
    """
    A sequential application of the input_x_gradient rule and the uniform rule to distribute the relevance uniformly to all inputs according to the uniform rule 
    as discussed in Section 3.3.2. 'Handling Matrix-Multiplication' of the paper 'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'.

    If one of the inputs is a constant or does not require gradients, no relevance is distributed to it.

    Parameters:
    -----------
    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch
    epsilon: float
        Small value to stabilize the denominator
    *inputs: at least one torch.Tensor
        The input tensors to the function
    """
        
    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):
    
        inputs, outputs = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        relevance_norm = out_relevance[0] / _stabilize(outputs, ctx.epsilon, inplace=False)
        relevance_norm = relevance_norm / len(inputs)

        # computes vector-jacobian product
        grads = torch.autograd.grad(outputs, inputs, relevance_norm)

        # return relevance at requires_grad indices else None
        return (None, None) + tuple(grads[i].mul_(inputs[i]) if ctx.requires_grads[i] else None for i in range(len(ctx.requires_grads)))



class TaylorDecompositionRule(WrapModule):
    """
    Generalized Taylor Decomposition with or without bias for any differentiable function according to the Equation 4-5 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Note: All inputs must be tensors and will receive relevance!

    Parameters:
    -----------
    module: nn.Module
        The module to be wrapped
    ref: iterable of torch.Tensor
        The reference point for the Jacobian computation
    bias: bool
        Whether to include the bias term in the relevance computation
    distribute_bias: callable
        A function to distribute the bias relevance to the input tensors, only used if bias=True
    """

    def __init__(self, module, ref=0, bias=False, distribute_bias=None):
        super(TaylorDecompositionRule, self).__init__(module)
        self.ref = ref
        self.bias = bias
        self.distribute_bias = distribute_bias

    def forward(self, *inputs):

        return taylor_decomposition_fn.apply(self.module, self.ref, self.bias, self.distribute_bias, *inputs)



class taylor_decomposition_fn(Function):
    """
    Generalized Taylor Decomposition with or without bias for any differentiable function according to the Equation 4-5 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    All inputs must be tensors and will receive relevance. If you want to exclude a tensor from the relevance computation, you must wrap the function accordingly.

    Parameters:
    -----------
    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch
    ref: iterable of torch.Tensor
        The reference point for the Jacobian computation
    bias: bool
        Whether to include the bias term in the relevance computation
    distribute_bias: callable
        A function to distribute the bias relevance to the input tensors, only used if bias=True
    *inputs: all torch.Tensor
        The input tensors to the function
    """

    @staticmethod
    def forward(ctx, fn, ref, bias, distribute_bias, *inputs):
        
        output = fn(*inputs)
        ctx.save_for_backward(*inputs)
        ctx.fn, ctx.ref, ctx.bias = fn, ref, bias
        ctx.distribute_bias = distribute_bias

        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs = ctx.saved_tensors

        if not ctx.bias:
            # compute jacobian at reference point and multiply from right side with inputs
            # bias term is omitted in this way
            _, Jvs = jvp(ctx.fn, ctx.ref, inputs)
            output = Jvs
        
        normed_relevance = out_relevance[0] / _stabilize(output, inplace=True)

        # compute jacobian at reference and multiply from left side with R/output
        _, vjpfunc = vjp(ctx.fn, *ctx.ref)
        grads = vjpfunc(normed_relevance)
        
        relevances = tuple(grads[i].mul_(inputs[i]) for i in range(len(inputs)))

        if ctx.bias and callable(ctx.distribute_bias):
            relevances = ctx.distribute_bias(inputs, relevances)

        # multiply vJ with reference point
        return (None, None, None) + relevances
    

class UniformRule(WrapModule):
    """
    Distribute the relevance uniformly to all inputs according to the uniform rule in Equation 7 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    Parameters:
    -----------
    module: nn.Module
        The module to be wrapped
    """

    def forward(self, *inputs):

        return uniform_rule_fn.apply(self.module, *inputs)


class uniform_rule_fn(Function):
    """
    Distribute the relevance uniformly to all inputs according to the uniform rule in Equation 7 of the paper:
    AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers

    Parameters:
    -----------
    fn: callable
        The function to be called with the inputs, must be differentiable in PyTorch
    *inputs: all torch.Tensor
        The input tensors to the function
    """

    @staticmethod
    def forward(ctx, fn, *inputs):

        output = fn(*inputs)
        ctx.save_for_backward(*inputs)

        return output

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs = ctx.saved_tensors
        n = len(inputs)
        return (None,) + tuple(out_relevance[0] / n for _ in range(n))