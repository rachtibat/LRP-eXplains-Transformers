.. _latent_feature_attribution_efficient:

Latent Feature Attribution
==========================

Since we get relevance values for each neuron in the model as a by-product, we know exactly how important each neuron is for the prediction of the model. 
Combined with Activation Maximization, we can label neurons in LLMs and even steer the generation process of the LLM by activating specialized knowledge neurons in latent space.

Tracing internal relevance flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can trace how relevance flows through the model using standard PyTorch hooks! We only need the activations * gradients, which we can obtain at any layer in the model!

To trace e.g. the relevance at the residual stream in a LLaMA model, we can attach a forward hook at the decoder layer:

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers.models.llama import modeling_llama
    import matplotlib.pyplot as plt
    import numpy as np

    from lxt.efficient import monkey_patch
    monkey_patch(modeling_llama, verbose=True)


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

    def hook_hidden_activation(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        
        # save the activation and make sure the gradient is also saved in the .grad attribute after the backward pass
        module.output = output
        module.output.retain_grad() if module.output.requires_grad else None

    model = modeling_llama.LlamaForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', device_map='cuda', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')

    # optional gradient checkpointing to save memory (2x forward pass)
    model.train()
    model.gradient_checkpointing_enable()

    # deactive gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    # apply hooks
    for layer in model.model.layers:
        layer.register_forward_hook(hook_hidden_activation)

    # forward & backard pass
    prompt_response = f"I have 5 cats and 3 dogs. My cats love to play with my"
    input_ids = tokenizer(prompt_response, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)

    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
    max_logits.backward(max_logits)

    print("Prediction:", tokenizer.convert_ids_to_tokens(max_indices))

    # trace relevance through layers
    relevance_trace = []
    for layer in model.model.layers:
        relevance = (layer.output * layer.output.grad).float().sum(-1).detach().cpu()
        # normalize relevance at each layer between -1 and 1
        relevance = relevance / relevance.abs().max()
        relevance_trace.append(relevance)

    relevance_trace = torch.cat(relevance_trace, dim=0)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    save_heatmap(relevance_trace.numpy().T, tokens, (20, 10), f"Latent Relevance Trace (Normalized)", f'latent_rel_trace.png')


Below we see the relevance trace of the residual stream in the LLaMA 3.1 8b model. 
We see that the model uses in the first layers the 'begin of text' token, the 'cats' token and also the 'dogs' token. The last token 'my' has the highest
relevance score, because it functions as query in the attention layers where the final prediction is writting into.

In later layers, the model converged to a final prediction and does not use any other tokens anymore. Interestingly, the 'begin of text' token has negative relevance i.e.
the representations encoded in the 'begin of text' token are decreasing the output logit of the prediction 'dogs'. This is hinting to the fact, that the model might
use the start token as a scratch pad.

In the spirit of Mechanistic Interpretability, the relevance score could be used to understand how these tokens interact with each other during the inference process.

.. raw:: html

    <embed src="_static/llama3.1_latent_rel_trace.png" width="600">

