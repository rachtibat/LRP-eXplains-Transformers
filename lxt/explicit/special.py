import math
import torch
import torch.fx
import torch.nn.functional as F
import lxt.explicit.rules as rules


@torch.no_grad()
def _prepare_key_padding_mask(mask, attn_mask, query):
    """
    Prepare the key padding mask for the attention operation.
    """
    # -- broadcast mask
    assert mask.ndim > 1 # [..., SeqLen]
    if mask.ndim == 2: # [Batch, ... , ... , SeqLen]
        b, k_len = mask.shape
        mask = mask.view(b, 1, 1, k_len)

    return F._canonical_mask(mask, "key_padding_mask", F._none_or_dtype(attn_mask), "attn_mask", query.dtype)

@torch.no_grad()
def _prepare_attn_mask(mask, query):
    """
    Prepare the attention mask for the attention operation.
    """
    # -- broadcast mask
    assert mask.ndim >= 2 # [..., SeqLen, SeqLen]
    if mask.ndim == 3: # [Batch * Heads, SeqLen, SeqLen]
        mask = mask.view(query.shape)

    return F._canonical_mask(mask, "attn_mask", None, "", query.dtype, False)

@torch.fx.wrap
def multi_head_attention_cp(query, key, value, batch_first, num_heads, head_dim, q_proj_weight, bias_q, k_proj_weight, bias_k, v_proj, out_proj,
                            key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
    """
    Implementing the CP-LRP (Conservative Propagation - LRP) rule for attention i.e. we don't let relevance flow through the softmax, but only through the value path. 
    This method *only works well in Vision Transformers* because here the advanced AttnLRP rules, which do use the softmax, have similar performance to CP-LRP rules. 
    The issue with AttnLRP is that using the softmax introduces gradient shattering, which requires applying the z-plus LRP rule. 
    This makes AttnLRP slightly less efficient and, based on our limited experiments, the small performance gain is not worthwhile in Vision Transformers.
    However, in Large Language Models, applying AttnLRP on the softmax is substantially better than CP-LRP and does not require the less efficient z-plus rule.
    Therefore, we choose the more efficient CP-LRP for attention and use AttnLRP for other parts of the ViT.

    Please refer to Section A.2.3. 'Tackling Noise in Vision Transformers' of the the paper
    'AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers'.

    Parameters:
    -----------
    query: torch.Tensor
        The query tensor of shape [SeqLen, Batch, Embed] if batch_first is False, otherwise [Batch, SeqLen, Embed]
    key: torch.Tensor
        The key tensor of shape [SeqLen, Batch, Embed] if batch_first is False, otherwise [Batch, SeqLen, Embed]
    value: torch.Tensor
        The value tensor of shape [SeqLen, Batch, Embed] if batch_first is False, otherwise [Batch, SeqLen, Embed]
    batch_first: bool
        Whether the input tensors are in batch_first format
    num_heads: int
        The number of attention heads
    head_dim: int
        The dimension of each attention head
    q_proj_weight: torch.Tensor
        The projection weight for the query tensor
    bias_q: torch.Tensor
        The bias for the query tensor
    k_proj_weight: torch.Tensor
        The projection weight for the key tensor
    bias_k: torch.Tensor
        The bias for the key tensor
    v_proj: torch.nn.Module
        The projection module for the value tensor
    out_proj: torch.nn.Module
        The projection module for the output tensor
    key_padding_mask: torch.Tensor
        The padding mask for the key tensor
    need_weights: bool
        Whether to return the attention weights
    attn_mask: torch.Tensor
        The attention mask
    average_attn_weights: bool
        Whether to average the attention weights
    
    Returns:
    --------
    out: torch.Tensor
        The output tensor of shape [SeqLen, Batch, Embed] if batch_first is False, otherwise [Batch, SeqLen, Embed]
    """

    

    if batch_first is False:
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

    batch_size, q_seq_length, embed_dim = query.shape
    _, v_seq_length, _ = value.shape

    # -- project inputs to new embedding
    with torch.no_grad():
        q = torch.nn.functional.linear(query, q_proj_weight, bias=bias_q)
        k = torch.nn.functional.linear(key, k_proj_weight, bias=bias_k)
    v = v_proj(value)

    # -- reshape for multiheadattention
    q = q.view(batch_size, q_seq_length, num_heads, head_dim)
    k = k.view(batch_size, v_seq_length, num_heads, head_dim)
    v = v.view(batch_size, v_seq_length, num_heads, head_dim)

    q = q.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Embed]
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # -- perform attention on each head
    with torch.no_grad():
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

        mask = torch.zeros_like(attn_logits).to(attn_logits)
        if key_padding_mask is not None:
            mask += _prepare_key_padding_mask(key_padding_mask, attn_mask, q)
        if attn_mask is not None:
            mask += _prepare_attn_mask(attn_mask, q)

        attn_logits = attn_logits + mask
        attention = torch.softmax(attn_logits, -1)

    y = rules.epsilon_lrp(torch.matmul, 1e-6, attention.detach(), v)

    # -- out projection
    y = y.permute(0, 2, 1, 3)
    y = y.reshape(batch_size, q_seq_length, embed_dim)
    out = out_proj(y)

    if batch_first is False:
        out = out.transpose(0, 1)

    if need_weights and average_attn_weights:
        return out, attention.mean(dim=1)
    elif need_weights:
        return out, attention
    else:
        return out, None