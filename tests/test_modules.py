import pytest
import math
import torch
import torch.nn as nn
import lxt.explicit.modules as lm
import lxt.explicit.special as ls
import lxt.explicit.rules as rules
from lxt.explicit.core import Composite

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


def test_MultiheadAttention():

    layer_gt = nn.MultiheadAttention(512, 4, batch_first=True)

    layer_lxt = lm.INIT_MODULE_MAPPING[lm.MultiheadAttention_CP](layer_gt, lm.MultiheadAttention_CP)

    ### test forward pass
    x = torch.randn(4, 12, 512)
    attn_mask = torch.randn(12, 12)

    y_gt, attn_gt = layer_gt(x, x, x, attn_mask=attn_mask)
    y_lxt, attn_lxt = layer_lxt(x, x, x, attn_mask=attn_mask)

    assert torch.allclose(y_gt, y_lxt, atol=1e-5, rtol=0)
    assert torch.allclose(attn_gt, attn_lxt, atol=1e-5, rtol=0)

    ### test backward pass
    def attribute_MLH_ground_truth(x, attn_mask):
        
        batch_size, q_seq_length, embed_dim = x.shape

        q = torch.nn.functional.linear(x, layer_lxt.q_proj_weight, bias=layer_lxt.bias_q)
        k = torch.nn.functional.linear(x, layer_lxt.k_proj_weight, bias=layer_lxt.bias_k)
        v = torch.nn.functional.linear(x, layer_lxt.v_proj.weight, bias=layer_lxt.v_proj.bias)

        # -- reshape for multiheadattention
        q = q.view(batch_size, q_seq_length, layer_lxt.num_heads, layer_lxt.head_dim)
        k = k.view(batch_size, q_seq_length, layer_lxt.num_heads, layer_lxt.head_dim)
        v = v.view(batch_size, q_seq_length, layer_lxt.num_heads, layer_lxt.head_dim)

        q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

        mask = torch.zeros_like(attn_logits).to(attn_logits)
        if attn_mask is not None:
            mask += ls._prepare_attn_mask(attn_mask, q)

        attn_logits = attn_logits + mask
        attention = torch.softmax(attn_logits, -1)

        y = torch.matmul(attention.detach(), v)

        # -- out projection
        y = y.permute(0, 2, 1, 3)
        y = y.reshape(batch_size, q_seq_length, embed_dim)
        out = torch.nn.functional.linear(y, layer_lxt.out_proj.weight, bias=layer_lxt.out_proj.bias)

        return out
    
    # apply epsilon rule on whole MHA function with softmax.detach() -> should be equal to lm.MultiheadAttention_CP
    x = torch.randn(4, 12, 512, requires_grad=True)
    y_gt = rules.epsilon_lrp_fn.apply(attribute_MLH_ground_truth, 1e-6, x, attn_mask)

    y_gt.backward(y_gt)
    rel_gt = x.grad
    x.grad = None

    Composite({
        lm.LinearInProjection: rules.EpsilonRule,
        lm.LinearOutProjection: rules.EpsilonRule,
    }).register(layer_lxt)

    y_lxt, _ = layer_lxt(x, x, x, attn_mask=attn_mask)

    y_lxt.backward(y_lxt)
    rel_lxt = x.grad
    assert torch.allclose(rel_gt, rel_lxt, rtol=0, atol=1e-1)

    # compute cosine similarity
    rel_gt = rel_gt.flatten()
    rel_lxt = rel_lxt.flatten()

    cos_sim = torch.dot(rel_gt, rel_lxt) / (torch.norm(rel_gt) * torch.norm(rel_lxt))
    assert cos_sim > 0.99




if __name__ == "__main__":

    test_Linear()
    test_LayerNorm()
    test_MultiheadAttention()

    print("ALL TESTS PASSED")
