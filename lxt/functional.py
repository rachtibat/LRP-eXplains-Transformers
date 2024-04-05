import torch
import torch.fx
from torch.autograd import Function
import torch.nn.functional as F


####################
### Sanity Check ###
####################

CONSERVATION_CHECK_FLAG = [False]

def conservation_check_wrap(func):
    #TODO: bug in add2_fn
    """
    Decorator to enable or disable the sanity check for LRP operations, i.e. testing if the LRP conservation property holds for all operations excluding bias terms.
    If the sanity check is enabled, the relevance is distributed uniformly to the input tensors, else the relevance is returned as computed by the function.
    This check is useful to verify if the operations used in the model are all LRP-compatible.
    """
    def wrapped(ctx, *out_relevance):

        inp_relevance = func(ctx, *out_relevance)

        if CONSERVATION_CHECK_FLAG[0]:

            out_rel_sum = sum(r.float().sum() for r in out_relevance if r is not None)
            inp_elements = sum(r.numel() for r in inp_relevance if r is not None)
            inp_rel_mean = out_rel_sum/inp_elements

            if torch.isnan(inp_rel_mean).any():
                raise ValueError(f"NaN at {func}")
            
            inp_relevance = tuple(torch.full(r.shape, inp_rel_mean).to(r.device) if r is not None else None for r in inp_relevance)


        return inp_relevance
        
    return wrapped

#####################
### LRP FUNCTIONS ###
#####################

@torch.fx.wrap
def add2(input_a, input_b, inplace=False, epsilon=1e-6):
    """
    Standard Epsilon-LRP rule for elementwise addition (along all dimensions) of two tensors according to the Equation 8 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    epsilon: float
        Small value to stabilize the denominator
    """
    
    return add2_tensors_fn.apply(input_a, input_b, inplace, epsilon)
    # return add2_fn_OLD.apply(input_a, input_b, inplace, epsilon)

@torch.fx.wrap
def softmax(input, dim, dtype=None, inplace=False):
    """
    Computes Relevance using Deep Taylor Decomposition at x (with bias) according to Proposition 3.1 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor
    dim: int
        The dimension to apply the softmax function
    dtype: torch.dtype
        Convert the input to this dtype before applying the softmax function
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    """
    return softmax_fn.apply(input, dim, dtype, inplace)

@torch.fx.wrap
def linear_epsilon(input, weight, bias=None, epsilon=1e-6):
    """
    Standard Epsilon-LRP rule for nn.functional.linear according to the Equation 8 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'
    or Equation 16 of the paper:
    On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor
    weight: torch.Tensor
        The weight tensor
    bias: torch.Tensor
        The bias tensor
    epsilon: float
        Small value to stabilize the denominator
    """
    return linear_epsilon_fn.apply(input, weight, bias, epsilon)

@torch.fx.wrap
def matmul(input_a, input_b, inplace=False, epsilon=1e-6):
    """
    Computes relevance by sequential application of the epsilon- and uniform-LRP rule according to Proposition 3.3 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    epsilon: float
        Small value to stabilize the denominator
    """
    return matmul_fn.apply(input_a, input_b, inplace, epsilon)

@torch.fx.wrap
def rms_norm_identity(hidden_states, weight, variance_epsilon):
    """
    Computes relevance for the LlamaRMSNorm layer according to Proposition 3.4 and Equation 9 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Since we apply the identity rule also to weight * hidden_states.to(input_dtype), we can apply the identity rule to the whole layer, i.e.
    distributing the relevance 100% to the input.

    Parameters:
    -----------
    hidden_states: torch.Tensor
        The input tensor
    weight: torch.Tensor
        The weight tensor
    variance_epsilon: float
        Small value to stabilize the denominator
    """
    return rms_norm_identity_fn.apply(hidden_states, weight, variance_epsilon)

@torch.fx.wrap
def mul2(input_a, input_b, inplace=False):
    """
    Uniform LRP rule for elementwise multiplication (along all dimensions) of two tensors according to Proposition 3.2 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    If one of the inputs is a constant or does not require gradients, the relevance is distributed 100% to the other input.

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    """
    return mul2_fn.apply(input_a, input_b, inplace)


###############################
### AUTOGRAD IMPLEMENTATION ###
###############################

def _stabilize(input, epsilon=1e-6, inplace=False):
    """
    Stabilize the input by adding a small value to it
    """
    if inplace:
        return input.add_(epsilon)
    else:
        return input + epsilon
    

class softmax_fn(Function):
    """
    Computes Relevance using Deep Taylor Decomposition at x (with bias) according to Proposition 3.1 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor
    dim: int
        The dimension to apply the softmax function
    dtype: torch.dtype
        Convert the input to this dtype before applying the softmax function
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    """

    @staticmethod
    def forward(ctx, inputs, dim, dtype=None, inplace=False):

        if dtype is not None:
            inputs = inputs.to(dtype)
    
        outputs = F.softmax(inputs, dim=dim, dtype=dtype)

        ctx.save_for_backward(inputs, outputs)
        ctx.inplace = inplace

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, output = ctx.saved_tensors

        if ctx.inplace:
            relevance = (out_relevance[0].sub_(output.mul_(out_relevance[0].sum(-1, keepdim=True)))).mul_(inputs)
        else:
            relevance = inputs * (out_relevance[0] - output * out_relevance[0].sum(-1, keepdim=True))
        
        return (relevance, None, None, None)


class linear_epsilon_fn(Function):
    """
    Standard Epsilon-LRP rule for nn.functional.linear according to the Equation 8 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'
    or Equation 16 of the paper:
    'On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation'

    Parameters:
    -----------
    input: torch.Tensor
        The input tensor
    weight: torch.Tensor
        The weight tensor
    bias: torch.Tensor
        The bias tensor
    epsilon: float
        Small value to stabilize the denominator
    """

    @staticmethod
    def forward(ctx, inputs, weight, bias=None, epsilon=1e-6):
        
        outputs = F.linear(inputs, weight, bias)
        ctx.save_for_backward(inputs, weight, outputs)
        ctx.epsilon = epsilon
    
        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        inputs, weight, outputs = ctx.saved_tensors
        epsilon = ctx.epsilon

        relevance_norm = out_relevance[0] / _stabilize(outputs, epsilon)

        relevance = torch.matmul(relevance_norm, weight).mul_(inputs)
        
        return (relevance, None, None, None)


class matmul_fn(Function):
    """
    Computes relevance by sequential application of the epsilon- and uniform-LRP rule according to Proposition 3.3 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    epsilon: float
        Small value to stabilize the denominator
    """
    
    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False, epsilon=1e-6):
        
        outputs = torch.matmul(input_a, input_b)
        ctx.save_for_backward(input_a, input_b, outputs)
        ctx.inplace, ctx.epsilon = inplace, epsilon

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        input_a, input_b, outputs = ctx.saved_tensors
        inplace, epsilon = ctx.inplace, ctx.epsilon

        if inplace:
            relevance_norm = out_relevance[0].div_(_stabilize(outputs.mul_(2), epsilon, inplace))
        else:
            relevance_norm = out_relevance[0] / _stabilize(outputs * 2, epsilon, inplace)

        relevance_a = torch.matmul(relevance_norm, input_b.transpose(-1, -2)).mul_(input_a)
        relevance_b = torch.matmul(input_a.transpose(-1, -2), relevance_norm).mul_(input_b)
        
        return (relevance_a, relevance_b, None, None)



class add2_tensors_fn(Function):
    """
    Standard Epsilon-LRP rule for elementwise addition (along all dimensions) of two tensors according to the Equation 8 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    epsilon: float
        Small value to stabilize the denominator
    """
    
    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False, epsilon=1e-6):
    
        outputs = input_a + input_b
        if any([inp.requires_grad for inp in (input_a, input_b)]):
            ctx.save_for_backward(input_a, input_b)
            ctx.epsilon, ctx.inplace = epsilon, inplace

        return outputs

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        #TODO: replace for conservation check with requires grad stuff

        input_a, input_b = ctx.saved_tensors

        if ctx.inplace:
            relevance_norm = out_relevance[0].div_(_stabilize(input_a + input_b, epsilon=ctx.epsilon, inplace=True))

            relevance_a = relevance_norm * input_a
            relevance_b = relevance_norm.mul_(input_b)

        else:
            relevance_norm = out_relevance[0] / _stabilize(input_a + input_b, epsilon=ctx.epsilon, inplace=True)

            relevance_a = relevance_norm * input_a
            relevance_b = relevance_norm * input_b

        return (relevance_a, relevance_b, None, None)



class rms_norm_identity_fn(Function):
    """
    Computes relevance for the LlamaRMSNorm layer according to Proposition 3.4 and Equation 9 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    Since we apply the identity rule also to weight * hidden_states.to(input_dtype), we can apply the identity rule to the whole layer, i.e.
    distributing the relevance 100% to the input.

    Parameters:
    -----------
    hidden_states: torch.Tensor
        The input tensor
    weight: torch.Tensor
        The weight tensor
    variance_epsilon: float
        Small value to stabilize the denominator
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, variance_epsilon):

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

        return weight * hidden_states.to(input_dtype)

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        return out_relevance + (None, None)


class mul2_fn(Function):
    """
    Uniform LRP rule for elementwise multiplication (along all dimensions) of two tensors according to Proposition 3.2 of the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'

    If one of the inputs is a constant or does not require gradients, the relevance is distributed 100% to the other input.

    Parameters:
    -----------
    input_a: torch.Tensor
        The first input tensor
    input_b: torch.Tensor
        The second input tensor
    inplace: bool
        Whether to perform the operation in place during the backward pass, will overwrite the relevance at the output
    """


    @staticmethod
    def forward(ctx, input_a, input_b, inplace=False):

        ctx.requires_grads = [i for i, inp in enumerate((input_a, input_b)) if isinstance(inp, torch.Tensor) and inp.requires_grad]
        ctx.inplace = inplace

        return input_a * input_b

    @staticmethod
    @conservation_check_wrap
    def backward(ctx, *out_relevance):

        n_required = len(ctx.requires_grads)

        if ctx.inplace:
            out_relevance = out_relevance[0].div_(n_required)
        else:
            out_relevance = out_relevance[0] / n_required

        # only return relevance at requires_grad indices else None
        return tuple(out_relevance if i in ctx.requires_grads else None for i in range(2)) + (None,)