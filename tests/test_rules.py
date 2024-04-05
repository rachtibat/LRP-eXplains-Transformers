import pytest
import torch
import lxt.rules as rules
import lxt.functional as functional
from torch.nn import functional as F
import torch.nn as nn

def test_epsilon_rule():

    x = torch.randn(1, 5, requires_grad=True)
    weights = torch.randn(5, 5, requires_grad=False)
    bias = torch.randn(5, requires_grad=False)

    init_relevance = torch.randn(1, 5, requires_grad=True)

    y_gt = functional.linear_epsilon.apply(x, weights, bias)
    relevance_gt, = torch.autograd.grad(y_gt, x, init_relevance)

    y_lxt = rules.EpsilonRule.apply(F.linear, x, weights, bias)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)

    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-5)


