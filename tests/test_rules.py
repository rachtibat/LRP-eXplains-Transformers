import pytest
import torch
import lxt.explicit.rules as rules
import lxt.explicit.functional as lf
from torch.nn import functional as F
import torch.nn as nn
from functools import partial

def test_epsilon_rule():

    x = torch.randn(1, 5, requires_grad=True)
    weights = torch.randn(5, 5, requires_grad=False)
    bias = torch.randn(5, requires_grad=False)

    init_relevance = torch.randn(1, 5, requires_grad=True)

    y_gt = lf.linear_epsilon(x, weights, bias)
    relevance_gt, = torch.autograd.grad(y_gt, x, init_relevance)

    layer = rules.EpsilonRule(partial(F.linear, weight=weights, bias=bias))
    y_lxt = layer(x)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)

    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-3)


if __name__ == "__main__":
    
    test_epsilon_rule()
    
    print("ALL TESTS PASSED")


