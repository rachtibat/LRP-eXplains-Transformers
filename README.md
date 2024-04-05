<div align="center">
  <img src="docs/source/_static/lxt_logo.png" width="300"/>
  <p>Layer-wise Relevance Propagation (LRP) extended to handle attention layers in Large Language Models (LLMs) and Vision Transformers (ViTs)</p>
</div>

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org)
[![Read the Docs](https://img.shields.io/badge/-Docs-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white&link=https://inseq.org)](https://google.de)

#### 🔥 Faithful Attributions

Attention-aware LRP (AttnLRP) **outperforms** gradient- and perturbation-based methods, provides faithful attributions for the **entirety** of a black-box transformer model while scaling in computational complexitiy $O(1)$ and memory requirements $O(\sqrt{N})$ with respect to the number of layers.

#### 🔎 Latent Feature Attribution & Visualization
Since we get relevance values for each neuron in the model as a by-product, we know exactly how important each neuron is for the prediction of the model. Combined with Activation Maximization, we can label neurons in LLMs and even steer the generation process of the LLM by activating specialized knowledge neurons in latent space!

#### 📃 Paper
For understanding the math behind it, take a look at the [paper](https://arxiv.org/abs/2402.05602)!
```
@article{achtibat2024attnlrp,
  title={AttnLRP: Attention-aware Layer-wise Relevance Propagation for Transformers},
  author={Achtibat, Reduan and Hatefi, Sayed Mohammad Vakilzadeh and Dreyer, Maximilian and Jain, Aakriti and Wiegand, Thomas and Lapuschkin, Sebastian and Samek, Wojciech},
  journal={arXiv preprint arXiv:2402.05602},
  year={2024}
}
```

#### 📄 License
This project is licensed under the BSD-3 Clause License, which means that LRP is a patented technology that can only be used free of charge for personal and scientific purposes.

## Roadmap
⚠️ Because of the high community interest, we release a first version of LXT that might change in the future.

- [x] LRP for LLMs
- [ ] LRP for ViTs
- [ ] Latent Feature Visualization
- [ ] Perturbation Evaluation
- [ ] Tests
- [ ] Other baseline methods e.g. Attention Rollout


## Getting Started

### Installation

To install directly from PyPI using pip, write:

```shell
$ pip install lxt
```

To reproduce the experiments from the paper, install from a manually cloned repository: 

```shell
$ git clone https://github.com/rachtibat/LRP-for-Transformers
$ pip install ./lxt
```

Tested with ``transformers==4.39.3``, ``torch==2.2.2``, ``python==3.11``

### 💡 How does the code work?
Layer-wise Relevance Propagation is a rule-based backpropagation algorithm. This means, that we can implement LRP in a singular backward pass!
To achieve this, we have implemented [custom PyTorch autograd Functions](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html) for commonly used operations in transformers. These functions behave identically in the forward pass, but compute LRP attributions in the backward pass. To compute the $\varepsilon$-LRP rule for a linear function $y = W x + b$, you can simply write
```python
import lxt.functional as lf

y = lf.linear_epsilon(x.requires_grad_(), W, b)
y.backward(y)

relevance = x.grad
```

There are also "super-functions" that wrap an arbitrary nn.Module and compute LRP rules via automatic vector-Jacobian products! These rules are simple to attach to models:

```python
from lxt.core import Composite
import lxt.rules as rules

model = nn.Sequential(
  nn.Linear(10, 10),
  RootMeanSquareNorm(),
)

Composite({
  nn.Linear: rules.EpsilonRule,
  RootMeanSquareNorm: rules.IdentityRule,
}).register(model)

print(model)
```
<div align="left">
  <img src="docs/source/_static/terminal.png" width="400"/>
</div>


### 🚀 Quickstart with 🤗 (Tiny)LLaMA 2, T5 or Mixtral 8x7b
For a quick demo, we provide modified huggingface source code of popular LLMs, where we already replaced all operations with their equivalent LXT variant.


### 🤖 Fast Graph Manipulation with torch.fx 

So that you don't have to edit your source code, we exploited [torch.fx](https://pytorch.org/docs/stable/fx.html) to symbolically trace the model and replace layers and functions on-the-fly! This means that many 🤗 models are natively supported. However, not all models are symbolically tracable and *gradient-checkpointing doesn't work properly*, but we can still use torch.fx as a tool to guide us in the building process!

### 🕵️‍♂️ Debugging LXT
Applying LRP to new models can be tricky! We provide some basic debugging tools to help you out!

### ⚙️ Tuning Vision Transformers
ViTs are vulnerable to noisy attributions, hence we must denoise the heatmap.
We propose to use the $\gamma$-LRP rule, where the $\gamma$ parameter must be tuned to the model and dataset!

## Documentaion
A documentation of LXT is available, however we will add more in the coming months!
The Roadmap lists what is still missing.

## Contribution
Feel free to explore the code and experiment with different datasets and models. We encourage contributions and feedback from the community. We are especially grateful for providing support for new model architectures! 🙏


## Acknowledgements
The code is heavily inspired by [Zennit](https://github.com/chr5tphr/zennit), a tool for LRP attributions in PyTorch using hooks. Zennit is 100% compatible wit LXT and offers even more LRP rules 🎉 If you like LXT, consider also liking Zennit ;)
