import pytest
import torch
import lxt.functional as functional
from torch.nn import functional as F
import torch.nn as nn


def test_softmax():

    x = torch.randn(16, 10, 32, requires_grad=True)
    init_relevance = torch.randn(16, 10, 32, requires_grad=True)

    y_gt = F.softmax(x, -1)

    # implement Proposition 3.1 of AttnLRP paper
    relevance_gt = x.float() * (init_relevance - y_gt * init_relevance.sum(-1, keepdim=True))

    # test inplace=False
    y_lxt = functional.softmax.apply(x, -1, torch.float32, False)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)
    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-5)

    # test inplace=True
    y_lxt = functional.softmax.apply(x, -1, torch.float32, True)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)
    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-5)

    print("softmax test passed")


def test_matmul():

    epsilon = 1e-9

    a = torch.randn(16, 10, 32, requires_grad=True)
    b = torch.randn(16, 32, 5, requires_grad=True)

    init_relevance = torch.randn(16, 10, 5, requires_grad=True)

    y_gt = torch.matmul(a, b)

    # implement Proposition 3.3 of AttnLRP paper
    relevance_a_gt = torch.einsum("bji, bip, bjp -> bji", a, b, init_relevance / (2*y_gt + epsilon))
    relevance_b_gt = torch.einsum("bji, bip, bjp -> bip", a, b, init_relevance / (2*y_gt + epsilon))

    # test inplace=False
    y_lxt = functional.matmul.apply(a, b, False, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)
    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-5)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-5)

    # test inplace=True
    y_lxt = functional.matmul.apply(a, b, True, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)
    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-5)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-5)


def test_linear():

    epsilon = 1e-9

    x = torch.randn(16, 10, requires_grad=True)
    bias = torch.randn(5, requires_grad=False)
    weight = torch.randn(5, 10, requires_grad=True)

    init_relevance = torch.randn(16, 5, requires_grad=True)

    y_gt = F.linear(x, weight, bias)

    # implement Equation 8 of AttnLRP paper
    relevace_gt = torch.einsum("ji, bi, bj -> bi", weight, x, init_relevance / (y_gt + epsilon))

    # test inplace=False
    y_lxt = functional.linear_epsilon.apply(x, weight, bias, epsilon)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)

    assert torch.allclose(relevace_gt, relevance_lxt, rtol=0, atol=1e-5)


def test_sum():

    epsilon = 1e-9

    a = torch.randn(16, 10, 32, requires_grad=True)
    b = torch.randn(16, 10, 32, requires_grad=True)

    init_relevance = torch.randn(16, 10, 32, requires_grad=True)

    y_gt = a + b

    # implement epsilon rule for summation
    relevance_a_gt = a * (init_relevance / (y_gt + epsilon))
    relevance_b_gt = b * (init_relevance / (y_gt + epsilon))

    # test inplace=False
    y_lxt = functional.add2.apply(a, b, False, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)

    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-5)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-5)

    # test inplace=True
    y_lxt = functional.add2.apply(a, b, True, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)

    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-5)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-5)