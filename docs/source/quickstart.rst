.. _quickstart:

Quickstart
======================
In this tutorial, we will focus on the Input*Gradient formulation of Attention-aware Layer-wise Relevance Propagation (AttnLRP).
This implementation is identical to the original one introduced in the AttnLRP paper, but more efficient and leverages the automatic differentiation capabilities of PyTorch more natively.
To understand how it works under the hood, please refer to the :ref:`under_the_hood_efficient` section.

LLaMA
~~~~~
You find the complete code in the example directory: `examples/llama.py`

**Step 1: Install & Import Required Libraries**

Before we start, ensure that you have the necessary libraries installed.

.. code-block:: bash

    pip install lxt

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers.models.llama import modeling_llama
    from lxt.efficient import monkey_patch
    from lxt.utils import pdf_heatmap, clean_tokens



**Step 2: Monkey Patch the LLaMA module**

To compute LRP in the backward pass, we need to modify the LLaMA module. Let's apply the monkey patch.

.. code-block:: python

    # Modify the LLaMA module to compute LRP in the backward pass
    monkey_patch(modeling_llama, verbose=True)


**Step 3: Load the Pre-trained LLaMA Model**

We'll load the LLaMA model and enable gradient checkpointing to save memory.

.. code-block:: python

    # Load the model
    path = 'meta-llama/Llama-3.2-1B-Instruct'
    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        path, 
        device_map='cuda', 
        torch_dtype=torch.bfloat16
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

LXT also works for quantized models! However, the relevances should be accumulated in ``torch.bfloat16`` to prevent numerical errors:

.. code-block:: python

    from transformers import BitsAndBytesConfig
        
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = modeling_llama.LlamaForCausalLM.from_pretrained(
        path,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )

**Step 4: Disable Gradients to save Memory & optionally enable Gradient Checkpointing**

To optimize memory usage, we'll deactivate gradients on the model parameters. Optionally, we activate gradient checkpointing, which will perform 2x forward and 1x backward passes.
We set the model into ``train()`` mode, because right now Huggingface does not allow to activate gradient checkpointing in ``eval()`` mode. 
(The monkey patch makes sure that ``nn.Dropout``'s rate is set to 0, which would be otherwise activated in ``train()`` mode.)

.. code-block:: python

    # Deactivate gradients on parameters
    for param in model.parameters():
        param.requires_grad = False

    # Optionally enable gradient checkpointing (2x forward pass)
    model.train()
    model.gradient_checkpointing_enable()


**Step 5: Forward Pass & Backward Pass**

We'll provide a context and a question for the model. Here's our example prompt.

.. code-block:: python

    # Define the prompt
    prompt = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. 
    There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) 
    and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, 
    Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. 
    As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed 
    due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. 
    As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route 
    from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the 
    North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had 
    climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: 
    George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether 
    they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, 
    using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. 
    The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the 
    north ridge on 25 May 1960. 
    Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

We compute now the gradients with respect to the input embeddings. PyTorch can't compute gradients for int64 tensors like ``inputs_ids``, hence we use the bfloat16 ``inputs_embeds``.


.. code-block:: python

    # Get input embeddings
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)


Make sure to activate gradient tracing for the input embeddings.

.. code-block:: python

    # Inference
    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

    # Take the maximum logit at last token position. You can also explain any other token, or several tokens together!
    max_logits, max_indices = torch.max(output_logits[0, -1, :], dim=-1)

    # Backward pass (the relevance is initialized with the value of max_logits)
    max_logits.backward()

    # Obtain relevance. (Works at any layer in the model!)
    relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()  # Cast to float32 for higher precision


Render Heatmaps in LaTeX
~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we normalize the relevance scores and visualize them inside a PDF.


.. code-block:: python

    # Normalize relevance between [-1, 1]
    relevance = relevance / relevance.abs().max()

    # Remove special characters that are not compatible wiht LaTeX
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    # Save heatmap as PDF
    pdf_heatmap(tokens, relevance[0], path='llama_heatmap.pdf', backend='xelatex')


.. raw:: html

    <embed src="_static/llama_heatmap.pdf" width="480" height="400" type="application/pdf">



BERT Classifier
~~~~~~~~~~~~~~~

Like in autoregressive generation, we can apply LXT to classification tasks by simply computing the gradient at the class logit.
Here, we use a pre-trained BERT model trained on the CoLA dataset. Each example is a sequence of words annotated with whether it is a grammatical *acceptable* or *unacceptable* English sentence.

Our grammatically incorrect sentence is:

``After five years of research, scientists concluded that transformer models work because they has lots of parameters and math stuff``

which has a mistake in the word "has" which should be "have".

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    import transformers.models.bert.modeling_bert as modeling_bert
    from lxt.utils import pdf_heatmap, clean_tokens
    from lxt.efficient import monkey_patch
    monkey_patch(modeling_bert, verbose=True)

    tokenizer = AutoTokenizer.from_pretrained("JeremiahZ/bert-base-uncased-cola")
    model = modeling_bert.BertForSequenceClassification.from_pretrained("JeremiahZ/bert-base-uncased-cola").to("cuda")

    for param in model.parameters():
        param.requires_grad = False

    # The mistake here is in the word "has" which should be "have"
    inputs = "After five years of research, scientists concluded that transformer models work because they has lots of parameters and math stuff."

    input_ids = tokenizer(inputs, return_tensors="pt", add_special_tokens=True).input_ids.to("cuda")
    inputs_embeds = model.bert.get_input_embeddings()(input_ids)

    logits = model(inputs_embeds=inputs_embeds.requires_grad_(True)).logits

    max_logits, max_indices = torch.max(logits, dim=-1)

    out = model.config.id2label[max_indices.item()]
    print("The label of the sequence is grammatically: ", out)

    max_logits.backward()

    relevance = (inputs_embeds * inputs_embeds.grad).float().sum(-1).detach().cpu()[0]

    relevance = relevance / relevance.abs().max()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)

    pdf_heatmap(tokens, relevance, path="heatmap_bert.pdf", backend="xelatex")

The heatmap is computed w.r.t. the highest prediction logit 'acceptable'. This means, the model made a *wrong* prediction, because it should be 'unacceptable'! So, the model has not the highest accuracy (:
However, looking at the heatmap, we still see that the word 'has' has a **negative** relevance score which indicates that it is **suppressing** the explained class 'acceptable'.
So, the model still believes 'has' is actually wrong, but it is not confident enough to predict the correct class.

.. raw:: html

    <embed src="_static/bert_heatmap.pdf" width="480" height="80" type="application/pdf">


GPT2 with Contrastive Explanations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While the GPT2 model is the most famous autoregressive model, it is also quite tricky to explain what many don't know actually.
The problem is that GPT2 outputs most of the time only negative logits, which might be an artefact of non-optimal training! For the training objective, this makes no difference, since the softmax
is invariant to a constant shift of the logits. You could still explain the model like in the previous examples, but in some edge cases (especially short prompts), the heatmap could be sign flipped.

To get rid of these negative logits, we found that contrastive explanations work quite well (without any official benchmarks yet) `Gu, et al. "Understanding individual decisions of cnns via contrastive backpropagation." Springer, 2018. <https://link.springer.com/chapter/10.1007/978-3-030-20893-6_8>`_
This is equivalent to explaining the softmax output instead of the logits. It works by initializing the chosen class logit with 1, and all others with -1/N, where N is the number of classes.

.. code-block:: python

    import torch
    from transformers import AutoTokenizer
    from transformers.models.gpt2 import modeling_gpt2

    from lxt.efficient import monkey_patch
    from lxt.utils import pdf_heatmap, clean_tokens

    monkey_patch(modeling_gpt2, verbose=True)

    model = modeling_gpt2.GPT2LMHeadModel.from_pretrained('openai-community/gpt2', device_map='cuda', torch_dtype=torch.bfloat16, attn_implementation="eager")

    # deactive gradients on parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

    prompt = """Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
    Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids)
    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits

    # the model predicts the wrong number, but we still explain it
    max_logit, max_index = torch.max(output_logits[0, -1, :], dim=-1)

    # --- contrastive explanation ---
    mask = torch.ones_like(output_logits[0, -1, :]) * -1 / output_logits[0, -1, :].size(-1)
    mask[max_index] = 1
    output_logits[0, -1, :].backward(mask)
    # -------------------------------

    relevance = (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()[0] 
    relevance = relevance / relevance.abs().max()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    tokens = clean_tokens(tokens)
    pdf_heatmap(tokens, relevance, path='heatmap_contrastive.pdf', backend='pdflatex')

.. raw:: html

    <embed src="_static/contrastive_heatmap.pdf" width="480" height="400" type="application/pdf">


The model predicts here the **wrong** class, and we still explain it. This shows, that GPT2 has not the best performance on this task, but the explanation is still meaningful.

If you don't want to use contrastive explanations, `Arras, et al. “Close Look at Decomposition-based XAI-Methods for Transformer Language Models.” arXiv preprint, 2025. <https://arxiv.org/abs/2502.15886>`_ recommends using the CP-LRP variant:

.. code-block:: python

    from lxt.efficient import monkey_patch
    from lxt.efficient.models.gpt2 import cp_LRP

    # apply CP-LRP instead of AttnLRP variant
    monkey_patch(modeling_gpt2, cp_LRP, verbose=True)


Vision Transformer
~~~~~~~~~~~~~~~~~~
Vision Transformers are susceptible to gradient shattering, which leads to very noisy heatmaps. 
Within the LRP framework, we have specialized rules that improve the signal-to-noise ratio and denoise the heatmaps.
One such rule is the Gamma rule. However, this rule requires to tune a gamma hyperparameter for each layer.
For simplicity, we select a few values that can be manually evaluated by looking at the heatmaps.

For that, we use the library ``zennit`` to define rules for the Conv2d and Linear layers, because LXT does not support the ``Gamma`` rule yet and
``zennit`` has more rules to choose from, e.g. ``ZPlus``, ``AlphaBeta``, ``Epsilon`` etc. Since ``zennit`` uses an explicit formulation of LRP (see :ref:`explicit_quickstart`),
we need to monkey patch to transform it into the Input*Gradient formulation.

Hence, please install

.. code-block:: bash

    pip install zennit


We start by patching the ``torchvision`` and ``zennit`` module:

.. code-block:: python

    import torch
    import itertools
    from PIL import Image
    from torchvision.models import vision_transformer

    from zennit.image import imgify
    from zennit.composites import LayerMapComposite
    import zennit.rules as z_rules

    from lxt.efficient import monkey_patch, monkey_patch_zennit

    monkey_patch(vision_transformer, verbose=True)
    monkey_patch_zennit(verbose=True)

Now we load the model, define a zennit composite and try out different values for the gamma hyperparameters.

.. code-block:: python

    def get_vit_imagenet(device="cuda"):
        """
        Load a pre-trained Vision Transformer (ViT) model with ImageNet weights.
        
        Parameters:
        device (str): Device to load the model on ('cuda' or 'cpu')
        
        Returns:
        tuple: (model, weights) - The ViT model and its pre-trained weights
        """
        weights =vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
        model = vision_transformer.vit_b_16(weights=weights)
        model.eval()
        model.to(device)
        
        # Deactivate gradients on parameters to save memory
        for param in model.parameters():
            param.requires_grad = False
            
        return model, weights

    # Load the pre-trained ViT model
    model, weights = get_vit_imagenet()

    # Load and preprocess the input image
    image = Image.open('docs/source/_static/cat_dog.jpg').convert('RGB')
    input_tensor = weights.transforms()(image).unsqueeze(0).to("cuda")

    # Store the generated heatmaps
    heatmaps = []

    # Experiment with different gamma values for Conv2d and Linear layers
    # Gamma is a hyperparameter in LRP that controls how much positive vs. negative
    # contributions are considered in the explanation
    for conv_gamma, lin_gamma in itertools.product([0.1, 0.25, 100], [0, 0.01, 0.05, 0.1, 1]):
        input_tensor.grad = None  # Reset gradients
        print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)
        
        # Define rules for the Conv2d and Linear layers using 'zennit'
        # LayerMapComposite maps specific layer types to specific LRP rule implementations
        zennit_comp = LayerMapComposite([
            (torch.nn.Conv2d, z_rules.Gamma(conv_gamma)),
            (torch.nn.Linear, z_rules.Gamma(lin_gamma)),
        ])
        
        # Register the composite rules with the model
        zennit_comp.register(model)
        
        # Forward pass with gradient tracking enabled
        y = model(input_tensor.requires_grad_())
        
        # Get the top 5 predictions
        _, top5_classes = torch.topk(y, 5, dim=1)
        top5_classes = top5_classes.squeeze(0).tolist()
        
        # Get the class labels
        labels = weights.meta["categories"]
        top5_labels = [labels[class_idx] for class_idx in top5_classes]
        
        # Print the top 5 predictions and their labels
        for i, class_idx in enumerate(top5_classes):
            print(f'Top {i+1} predicted class: {class_idx}, label: {top5_labels[i]}')
        
        # Backward pass for the highest probability class
        # This initiates the LRP computation through the network
        y[0, top5_classes[0]].backward()
        
        # Remove the registered composite to prevent interference in future iterations
        zennit_comp.remove()
        
        # Calculate the relevance by computing Input*Gradient
        # This is the final step of LRP to get the pixel-wise explanation
        heatmap = (input_tensor * input_tensor.grad).sum(1)
        
        # Normalize relevance between [-1, 1] for plotting
        heatmap = heatmap / abs(heatmap).max()
        
        # Store the normalized heatmap
        heatmaps.append(heatmap[0].detach().cpu().numpy())

    # Visualize all heatmaps in a grid (3×5) and save to a file
    # vmin and vmax control the color mapping range
    imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('vit_heatmap.png')

.. raw:: html

    <embed src="_static/cat_dog.jpg" width="480">

.. raw:: html

    <embed src="_static/cat_god_gamma_search_torch.png">