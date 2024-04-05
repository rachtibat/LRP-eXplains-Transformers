.. _quickstart:
Quickstart
==========

Layer-wise Relevance Propagation is a rule-based backpropagation algorithm. This means, that we can implement LRP in a singular backward pass!
In order to achieve this, we implemented `custom PyTorch autograd Functions <https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_ for commonly used operations in transformers. These functions behave identically in the forward pass, but compute LRP attributions in the backward pass. 

To understand how LXT works under the hood, you can read :ref:`under_the_hood`.

For a quick demo, we provide modified huggingface source-code for (Tiny)LLaMA 2, T5 and Mixtral 8x7b, where we replaced all torch operations with LRP-compatible operations.
You can find all models in ``lxt.models``.

It is recommended to install ``accelerate`` to load huge LLM weights.

.. code-block:: bash

    pip install accelerate


General Usage Pattern
~~~~~~~~~~~~~~~~~~~~~~

To obtain relevances with LXT, we have to attach LRP rules to the model and then compute a backward pass. The input token ids of LLMs are not differentiable, but the input embeddings are!
Hence, we must first convert the input ids to tensor embeddings, set the ``requires_grad`` attribute, run the backward pass and then obtain the relevances from the ``.grad``
attribute of the input embeddings. Optionally, we enable gradient checkpointing to trade compute for GPU RAM.

.. code-block:: python

    # load model
    model = Model()

    # (optionally enable gradient checkpointing)
    lxt_model.gradient_checkpointing_enable()

    # apply LXT to the model
    lxt_model = lrp.register(model)

    # load input_ids
    input_ids = tokenizer(prompt)
    
    # transform input_ids to tensor embeddings
    input_embeddings = lxt_model.embedding_layer(input_ids)
    input_embeddings.requires_grad = True

    # run inference 
    output_logits = lxt_model(input_embeddings)

    # select token to explain
    select_class_logit = output_logits[0, -1, :].max()

    # run backward
    select_class_logit.backward(select_class_logit)

    # obtain relevances by summing over embedding dimension i.e. keeping sequence dimension
    relevance = input_embeddings.grad.float().sum(-1)





Render Heatmaps in LaTeX
~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a tool to save attributions as LaTeX PDF files. For that, you must install ``pdflatex`` or preferable ``xelatex``
(which supports more characters).


.. code-block:: python

    from lxt.utils import pdf_heatmap, clean_words

    # convert token ids to strings
    words = tokenizer.convert_ids_to_tokens(input_ids[0])

    # removes the '_' character of tokens
    words = clean_words(words)

    # normalize relevance between [-1, 1] for plotting
    relevance = relevance / relevance.abs().max()

    # generate PDF file
    pdf_heatmap(words, relevance, path='heatmap.pdf', backend='xelatex')



TinyLLaMA
~~~~~~~~~~

TinyLLaMA is a `very small open-source model <https://github.com/jzhang38/TinyLlama>`_ that can be used for a quick demo.

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from lxt.models.llama import LlamaForCausalLM, attnlrp
    from lxt.utils import pdf_heatmap, clean_tokens

    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # apply AttnLRP rules
    attnlrp.register(model)

    prompt = """\
    Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
    Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)

    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

    max_logits.backward(max_logits)
    relevance = input_embeds.grad.float().sum(-1).cpu()[0]

    # normalize relevance between [-1, 1] for plotting
    relevance = relevance / relevance.abs().max()

    # remove '_' characters from token strings
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    pdf_heatmap(tokens, relevance, path='heatmap.pdf', backend='xelatex')

.. raw:: html

    <embed src="_static/attn_lrp_heatmap_tiny.pdf" width="480" height="400" type="application/pdf">


Generally, the contrast in the heatmap is further strengthened if the softmax output is also explained.
However, we didn't explore this in our paper.

.. code-block:: python

    import lxt.functional as lf

    # ...

    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    output = lf.softmax(output_logits, -1)
    max_logits, max_indices = torch.max(output[0, -1, :], dim=-1)

    max_logits.backward(max_logits)

    # ...

.. raw:: html

    <embed src="_static/attn_lrp_heatmap_tiny_softmax.pdf" width="480" height="400" type="application/pdf">

LLaMA 2
~~~~~~~

Like TinyLLaMA, we simply change the URL of the huggingface repository since both models share the same architecture.
It is recommended to enable gradient checkpointing to save GPU RAM.

.. code-block:: python

    from lxt.models.llama import LlamaForCausalLM, attnlrp

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="cuda")

    # enable gradient checkpointing
    model.gradient_checkpointing_enable()


Mixtral 8x7b  
~~~~~~~~~~~~~

LXT also works for quantized models, however the relevances should be accumulated in ``torch.bfloat16`` to prevent numerical errors.

.. note::
   You need approx. 30 GB of GPU RAM to run the model!

.. code-block:: python

    from transformers import BitsAndBytesConfig
    from lxt.models.mixtral import MixtralForCausalLM, attnlrp
        
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=quantization_config, device_map="auto", use_safetensors=True, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    attnlrp.register(model)

    # ...

    # this model benefits more then others from explaining the output softmax too
    output = lf.softmax(output_logits, -1)


Flan-T5  
~~~~~~~~

Coming soon ...


Vision Transformer
~~~~~~~~~~~~~~~~~~

Coming soon ...