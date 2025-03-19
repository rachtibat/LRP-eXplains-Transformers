.. _explicit_quickstart:

Quickstart for the Mathematical Explicit Version
================================================


.. important::
   Here, we discuss the slower, but mathematical explicit implementation of LRP. This is useful for understanding the inner workings of LRP and for debugging purposes.
   For a more versatile and faster implementation, we recommend using the efficient implementation discussed at :ref:`quickstart`.


Layer-wise Relevance Propagation is a rule-based backpropagation algorithm. This means, that we can implement LRP in a single backward pass!
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
    from lxt.explicit.models.llama import LlamaForCausalLM, attnlrp
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
For that, we have a temperature hyperparameter in the softmax that should be set to a value greater than 1 to prevent
that the softmax is too confident and hence the gradient vanishes (more details in the paper, Appendix A.2.4). 
However, we didn't explore this in our experiments.

.. code-block:: python

    import lxt.explicit.functional as lf

    # ...

    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    output = lf.softmax(output_logits, -1, temperature=2)
    max_logits, max_indices = torch.max(output[0, -1, :], dim=-1)

    max_logits.backward(max_logits)

    # ...

.. raw:: html

    <embed src="_static/attn_lrp_heatmap_tiny_softmax.pdf" width="480" height="400" type="application/pdf">

LLaMA 2/3
~~~~~~~~~

Like TinyLLaMA, we simply change the URL of the huggingface repository since TinyLLaMA, LLaMA 2 and LLaMA 3 share the same architecture.
It is recommended to enable gradient checkpointing to save GPU RAM.

.. code-block:: python

    from lxt.explicit.models.llama import LlamaForCausalLM, attnlrp

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
    from lxt.explicit.models.mixtral import MixtralForCausalLM, attnlrp
        
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", quantization_config=quantization_config, device_map="auto", use_safetensors=True, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    attnlrp.register(model)

    # ...



Vision Transformer: OpenCLIP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Vision Transformers are susceptible to gradient shattering, which leads to very noisy heatmaps. 
Within the LRP framework, we have specialized rules that improve the signal-to-noise ratio and denoise the heatmaps.
One such rule is the Gamma rule. However, this rule requires to tune a gamma hyperparameter for each layer.
For simplicity, we select a few values that can be manually evaluated by looking at the heatmaps.

In contrast to the examples above, we take here advantage of the torch.fx graph manipulation capabilities introduced in :ref:`on_the_fly`.
In ``lxt.models.openclip.attnlrp``, we define a set of functions that are present inside the OpenCLIP ViT-G-14 model and replace them with LXT-compatible functions (Take a look into it!).
Further, we use the library ``Zennit`` to define rules for the Conv2d and Linear layers, because LXT does not support the ``Gamma`` rule yet and
``Zennit`` has more rules to choose from, e.g. ``ZPlus``, ``AlphaBeta``, ``Epsilon`` etc.

Hence, please install

.. code-block:: bash

    pip install zennit
    pip install open_clip_torch

.. note::
   Graph tracing does not work for models that require gradient checkpointing at this moment!

.. code-block:: python

    import torch
    import open_clip
    import itertools
    from PIL import Image

    import lxt.explicit.functional as lf
    from lxt.explicit.models.openclip import attnlrp
    from zennit.composites import LayerMapComposite
    import zennit.rules as z_rules
    from zennit.image import imgify


    device = 'cuda'

    # Load the model and the tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    model.eval()
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    # Load an image and tokenize a text
    text = tokenizer(['a beautiful LRP heatmap', 'a dog', 'a cat']).to(device)
    image = preprocess(Image.open('docs/source/_static/cat_dog.jpg')).unsqueeze(0).to(device)

    # trace the model with a dummy input
    # verbose=True prints all functions/layers found and replaced by LXT
    # you will see at the last entry that e.g. tensor.exp() is not supported by LXT. This is not a problem in our case,
    # because this function is not used in the backward pass and therefore does not need to be replaced.
    # (look into the open_clip.transformer module code!)
    x = torch.randn(1, 3, 224, 224, device=device)
    traced = attnlrp.register(model, dummy_inputs={'image': x, 'text': text}, verbose=True)

    # for Vision Transformer, we must perform a grid search for the best gamma hyperparameters
    # in general, it is enough to concentrate on the Conv2d and MLP layers
    # for simplicity we just use a few values that can be evaluated by hand & looking at the heatmaps
    heatmaps = []
    for conv_gamma, lin_gamma in itertools.product([0.1, 0.5, 100], [0, 0.01, 0.05, 0.1, 1]):

        print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)

        # we define rules for the Conv2d and Linear layers using 'Zennit'
        zennit_comp = LayerMapComposite([
                (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
                (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
            ])

        # register composite
        zennit_comp.register(traced)

        # forward & backward pass
        y = traced(image.requires_grad_(True), text)
        logits = lf.matmul(y[0], y[1].transpose(0, 1))

        # explain the dog class ("a dog")
        image.grad = None
        logits[0, 1].backward()

        # normalize the heatmap
        heatmap = image.grad[0].sum(0)
        heatmap = heatmap / abs(heatmap).max()
        heatmaps.append(heatmap.cpu().numpy())

        # zennit composites can be removed so that we can register a new one!
        zennit_comp.remove()

    # save the heatmaps as a grid
    imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('heatmap.png')

.. raw:: html

    <embed src="_static/cat_dog.jpg" width="480">

.. raw:: html

    <embed src="_static/cat_dog_gamma_search.png">
