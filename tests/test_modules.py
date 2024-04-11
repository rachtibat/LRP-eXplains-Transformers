import pytest
import torch
import torch.nn as nn
import lxt.modules as lm

def test_LayerNorm():

    layer_gt = nn.LayerNorm(4092)
    layer_gt.weight = nn.Parameter(torch.randn(4092))
    layer_gt.bias = nn.Parameter(torch.randn(4092))

    layer_lxt = lm.INIT_MODULE_MAPPING[lm.LayerNormEpsilon](layer_gt, lm.LayerNormEpsilon)

    x = torch.randn(32, 500, 4092)

    y_gt = layer_gt(x)
    y_lxt = layer_lxt(x)

    assert torch.allclose(y_gt, y_lxt, atol=1e-5, rtol=0)

def test_Linear():

    layer_gt = nn.Linear(4092, 2048)
    layer_gt.weight = nn.Parameter(torch.randn(2048, 4092))
    layer_gt.bias = nn.Parameter(torch.randn(2048))

    layer_lxt = lm.INIT_MODULE_MAPPING[lm.LinearEpsilon](layer_gt, lm.LinearEpsilon)

    x = torch.randn(32, 500, 4092)

    y_gt = layer_gt(x)
    y_lxt = layer_lxt(x)

    assert torch.allclose(y_gt, y_lxt, atol=1e-5, rtol=0)


if __name__ == "__main__":

    test_Linear()
    test_LayerNorm()
