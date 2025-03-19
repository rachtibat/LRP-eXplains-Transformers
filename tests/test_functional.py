import pytest
import torch
import lxt.explicit.functional as lf
from torch.nn import functional as F
import torch.nn as nn


def test_softmax():

    x = torch.randn(16, 10, 32, requires_grad=True)
    init_relevance = torch.randn(16, 10, 32, requires_grad=True)

    y_gt = F.softmax(x, -1)

    # implement Proposition 3.1 of AttnLRP paper
    relevance_gt = x.float() * (init_relevance - y_gt * init_relevance.sum(-1, keepdim=True))

    # test inplace=False
    y_lxt = lf.softmax(x, -1, torch.float32, False)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)
    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-5)

    # test inplace=True
    y_lxt = lf.softmax(x, -1, torch.float32, True)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)
    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-5)


def test_matmul():

    epsilon = 1e-9

    a = torch.randn(2, 10, 32, requires_grad=True)
    b = torch.randn(2, 32, 5, requires_grad=True)

    init_relevance = torch.randn(2, 10, 5, requires_grad=True)

    y_gt = torch.matmul(a, b)

    # implement Proposition 3.3 of AttnLRP paper
    relevance_a_gt = torch.einsum("bji, bip, bjp -> bji", a, b, init_relevance / (2*y_gt + epsilon))
    relevance_b_gt = torch.einsum("bji, bip, bjp -> bip", a, b, init_relevance / (2*y_gt + epsilon))

    # test inplace=False
    y_lxt = lf.matmul(a, b, False, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)
    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-4)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-4)

    # test inplace=True
    y_lxt = lf.matmul(a, b, True, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)
    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-4)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-4)


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
    y_lxt = lf.linear_epsilon(x, weight, bias, epsilon)
    relevance_lxt, = torch.autograd.grad(y_lxt, x, init_relevance)

    assert torch.allclose(relevace_gt, relevance_lxt, rtol=0, atol=1e-3)


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
    y_lxt = lf.add2(a, b, False, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)

    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-4)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-4)

    # test inplace=True
    y_lxt = lf.add2(a, b, True, epsilon)
    relevance_a_lxt, relevance_b_lxt = torch.autograd.grad(y_lxt, (a, b), init_relevance)

    assert torch.allclose(relevance_a_gt, relevance_a_lxt, rtol=0, atol=1e-5)
    assert torch.allclose(relevance_b_gt, relevance_b_lxt, rtol=0, atol=1e-5)


def test_mean():

    epsilon = 1e-9

    a = torch.randn(1, 8, 32, requires_grad=True)
    init_relevance = torch.randn(1, 8)

    # implement epsilon rule for mean
    relevance_gt = a * (init_relevance.unsqueeze(-1) / (a.sum(-1).unsqueeze(-1) + epsilon))

    ## --- test keep_dim=True
    y_lxt = lf.mean(a, -1, True, epsilon)
    relevance_lxt, = torch.autograd.grad(y_lxt, a, init_relevance.unsqueeze(-1))

    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-4)

    ## --- test keep_dim=False
    y_lxt = lf.mean(a, -1, False, epsilon)
    relevance_lxt, = torch.autograd.grad(y_lxt, a, init_relevance)

    assert torch.allclose(relevance_gt, relevance_lxt, rtol=0, atol=1e-4)


def test_layernorm():

    x = torch.randn(1, 2, 8, requires_grad=True)
    init_relevance = torch.randn(1, 2, 8)

    layer = torch.nn.LayerNorm(8)
    weight = torch.randn_like(layer.weight)
    bias = torch.randn_like(layer.bias)
    layer.weight = nn.Parameter(weight)
    layer.bias = nn.Parameter(bias)
    layer.weight.requires_grad_(False)
    layer.bias.requires_grad_(False)

    ### ground truth
    y = lf.layer_norm(x, weight, bias, layer.eps)
    relevance_gt, = torch.autograd.grad(y, x, init_relevance)

    ### lxt
    y = lf._layer_norm_slower(x, weight, bias, layer.eps)
    relevance_lxt, = torch.autograd.grad(y, x, init_relevance)

    assert torch.allclose(relevance_lxt, relevance_gt, rtol=0, atol=1e-1)

    # compute cosine similarity
    rel_gt = relevance_gt.flatten()
    rel_lxt = relevance_lxt.flatten()

    cos_sim = torch.dot(rel_gt, rel_lxt) / (torch.norm(rel_gt) * torch.norm(rel_lxt))
    assert cos_sim > 0.99


def test_normalize():

    x = torch.randn(1, 4, 32, requires_grad=True)
    r_gt = torch.randn(1, 4, 32)

    weight, variance_epsilon = torch.randn(32), 1e-9
    y = lf.rms_norm_identity(x, weight, variance_epsilon)
    y.backward(r_gt)

    assert torch.allclose(x.grad, r_gt, rtol=0, atol=1e-5)
    x.grad.zero_()

    y = lf.normalize(x, p=2, dim=1)
    y.backward(r_gt)

    assert torch.allclose(x.grad, r_gt, rtol=0, atol=1e-5)


if __name__ == "__main__":

    test_softmax()
    test_matmul()
    test_linear()
    test_sum()
    test_mean()
    test_layernorm()
    test_normalize()

    print("ALL TESTS PASSED")