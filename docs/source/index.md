# Layer-wise Relevance Propagation eXplains Transformers (LXT)

Welcome to the Documentation! 
LXT contains the Layer-wise Relevance Propagation (LRP) implementation extended to handle attention layers in Large Language Models (LLMs) and Vision Transformers (ViTs).

Attention-aware LRP (AttnLRP) **outperforms** gradient- and perturbation-based methods, provides faithful attributions for the **entirety** of a black-box transformer model while scaling in computational complexitiy O(1) and memory requirements O(√N).


```{eval-rst} 
.. important::
   Project is under active development! Feature visualization and perturbation experiments are still missing.
```

::::{grid} 2

:::{card} 🚀 Quickstart
:link: quickstart
:link-type: ref
Examples for 🤗 (Tiny)LLaMA 2, T5 and Mixtral 8x7b
:::

:::{card} 💡 LXT Drop-In Replacements
:link: drop_in_replacement
:link-type: ref
Modify your custom source-code
:::

:::{card} 🤖 On-the-fly Modifications
:link: on_the_fly
:link-type: ref
Modify your code during runtime via LXT Composites and Symbolic Graph Tracing
:::

:::{card} 🕵️‍♂️ Debugging LXT
:link: debugging
:link-type: ref
How to check, if my LXT implementation makes sense?
:::

::::


## Installation

To install directly from PyPI using pip, write:

```shell
$ pip install lxt
```

To reproduce the experiments from the paper, install from a manually cloned repository: 

```shell
$ git clone https://github.com/rachtibat/LRP-for-Transformers
$ pip install ./lxt
```

## License
This project is licensed under the BSD-3 Clause License, which means that LRP is a patented technology that can only be used free of charge for personal and scientific purposes.

## Paper
For understanding the math behind it, take a look at the [paper](https://arxiv.org/abs/2402.05602)!

```latex
@article{achtibat2024attnlrp,
  title={AttnLRP: Attention-aware Layer-wise Relevance Propagation for Transformers},
  author={Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  journal={arXiv preprint arXiv:2402.05602},
  year={2024}
}
```


## Table of Content

```{eval-rst} 
.. toctree::
   quickstart
   under-the-hood
   drop-in-replacement
   on-the-fly
   debugging
   :maxdepth: 3
   :caption: LXT Usage:

.. toctree::
   feature-visualization
   :maxdepth: 3
   :caption: Experiments:

.. toctree::
   core
   functional
   modules
   rules
   :maxdepth: 2
   :caption: API Documentation:
```

