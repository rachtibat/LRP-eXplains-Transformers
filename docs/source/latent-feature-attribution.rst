.. _latent_feature_attribution:

Latent Feature Attribution
==========================

Since we get relevance values for each neuron in the model as a by-product, we know exactly how important each neuron is for the prediction of the model. 
Combined with Activation Maximization, we can label neurons in LLMs and even steer the generation process of the LLM by activating specialized knowledge neurons in latent space.

Tracing internal relevance flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can trace how relevance flows through the model using standard PyTorch backward hooks!
Since we define custom autograd functions, we replace the standard gradient with actual LRP attribution scores. This means, that PyTorch
works as usual, but the gradients are replaced with LRP scores!

To trace e.g. the relevance at the residual stream in a LLaMA model, we can attach a backward hook at the decoder layer:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from lxt.explicit.models.llama import LlamaForCausalLM, attnlrp
    import matplotlib.pyplot as plt
    import numpy as np

    def save_heatmap(values, tokens, figsize, title, save_path):
        fig, ax = plt.subplots(figsize=figsize)

        abs_max = abs(values).max()
        im = ax.imshow(values, cmap='bwr', vmin=-abs_max, vmax=abs_max)
        
        layers = np.arange(values.shape[-1])

        ax.set_xticks(np.arange(len(layers)))
        ax.set_yticks(np.arange(len(tokens)))

        ax.set_xticklabels(layers)
        ax.set_yticklabels(tokens)

        plt.title(title)
        plt.xlabel('Layers')
        plt.ylabel('Tokens')
        plt.colorbar(im)

        plt.show()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def hidden_relevance_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        module.hidden_relevance = output.detach().cpu()


    # load model & apply AttnLRP
    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model.eval()
    attnlrp.register(model)

    # apply hooks
    for layer in model.model.layers:
        layer.register_full_backward_hook(hidden_relevance_hook)

    # forward & backard pass
    prompt_response = f"<s>I have 5 cats and 3 dogs. My cats love to play with my"
    input_ids = tokenizer(prompt_response, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)

    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
    max_logits.backward(max_logits)

    print("Prediction:", tokenizer.convert_ids_to_tokens(max_indices))

    # trace relevance through layers
    relevance_trace = []
    for layer in model.model.layers:
        relevance = layer.hidden_relevance[0].sum(-1)
        # normalize relevance at each layer between -1 and 1
        relevance = relevance / relevance.abs().max()
        relevance_trace.append(relevance)

    relevance_trace = torch.stack(relevance_trace)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    save_heatmap(relevance_trace.float().numpy().T, tokens, (20, 10), f"Latent Relevance Trace (Normalized)", f'latent_rel_trace.png')


Below we see the relevance trace of the residual stream in the LLaMA model. 
We see that the model uses in the first layers the '<s>' token, the 'ats' token of the word 'cats' and also the '_dogs' token. The last token '_my' has the highest
relevance score, because it functions as query in the attention layers where the final prediction is writting into.

In later layers, the model converged to a final prediction and does not use any other tokens anymore. Interestingly, the '<s>' token has negative relevance in later layers i.e.
the representations encoded in the '<s>' token are decreasing the output logit of the prediction '_dogs'. This is hinting to the fact, that the model might
use the start token as a scratch pad.

In the spirit of Mechanistic Interpretability, the relevance score could be used to understand how these tokens interact with each other during the inference process.

.. raw:: html

    <embed src="_static/latent_rel_trace.png" width="600">

