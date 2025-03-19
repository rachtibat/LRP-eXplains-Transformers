# Layer-wise Relevance Propagation eXplains Transformers (LXT)

Welcome to the Documentation! 
LXT contains the Layer-wise Relevance Propagation (LRP) implementation extended to handle attention layers in Large Language Models (LLMs) and Vision Transformers (ViTs).

**🔥 Highly efficient & Faithful Attributions**

Attention-aware LRP (AttnLRP) **outperforms** gradient-, decomposition- and perturbation-based methods, provides faithful attributions for the **entirety** of a black-box transformer model while scaling in computational complexity O(1) and memory requirements O(√N) with respect to the number of layers.

**🔎 Latent Feature Attribution & Visualization**

Since we get relevance values for each single neuron in the model as a by-product, we know exactly how important each neuron is for the prediction of the model. Combined with Activation Maximization, we can label neurons or SAE features in LLMs and even steer the generation process of the LLM by activating specialized knowledge neurons in latent space!

**📚 Paper**

For the mathematical details and foundational work, please take a look at our paper:  
[Achtibat, et al. “AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.” ICML 2024.](https://proceedings.mlr.press/v235/achtibat24a.html) 


```{eval-rst} 
.. important::
   Project is under active development!
```

::::{grid} 2

:::{card} 🚀 Quickstart
:link: quickstart
:link-type: ref
Example for 🤗 LLaMA & many more 
:::

:::{card} 💡 Explicit Implementation
:link: explicit_quickstart
:link-type: ref
Using the mathematical explicit but slow version
:::

:::{card} 🛠️ Supported Models & Extending LXT
:link: extending
:link-type: ref
List of available models & add support for your own model
:::

:::{card} 🔎 Latent Feature Attribution
:link: latent_feature_attribution_efficient
:link-type: ref
Trace the internal reasoning process of a transformer
:::

::::


## Installation

To install directly from PyPI using pip, write:

```shell
$ pip install lxt
```

or install from the cloned GitHub repository:

```shell
$ git clone https://github.com/rachtibat/LRP-for-Transformers
$ pip install ./lxt
```

## License
This project is licensed under the BSD-3 Clause License, which means that LRP is a patented technology that can only be used free of charge for personal and scientific purposes.

## Citation
```latex
@InProceedings{pmlr-v235-achtibat24a,
  title = {{A}ttn{LRP}: Attention-Aware Layer-Wise Relevance Propagation for Transformers},
  author = {Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages = {135--168},
  year = {2024},
  editor = {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = {235},
  series = {Proceedings of Machine Learning Research},
  month = {21--27 Jul},
  publisher = {PMLR}
}
```


## Table of Content

```{eval-rst} 

.. toctree::
   quickstart
   under-the-hood-efficient
   extending
   latent-feature-attribution-efficient
   :maxdepth: 3
   :caption: Efficient:

.. toctree::
   explicit_quickstart
   under-the-hood
   drop-in-replacement
   on-the-fly
   latent-feature-attribution
   :maxdepth: 3
   :caption: Mathematical Explicit:
