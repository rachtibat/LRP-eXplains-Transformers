# README

Welcome to the `examples` folder of our project!

## Paper Examples

The `examples/paper` directory contains code used to generate the heatmaps presented in the original paper. This code uses an explicit but inefficient formulation of LRP. While this implementation is not recommended for real-world usage due to its inefficiency & complexity, it is very useful for understanding the mathematical principles behind LRP.

Tested with older version of `transformers==4.46.2`, `torch==2.1.0`, `python==3.11`.

## Efficient Implementation

For real-world applications, we recommend using the code here `examples/*` and not `examples/paper`. This implementation of Layer-wise Relevance Propagation (LRP) is more efficient and leverages the automatic differentiation capabilities of PyTorch more natively. It is implemented via a Gradient*Input framework, which is briefly described in the following paper: [A Close Look at Decomposition-based XAI-Methods for Transformer Language Models, Leila Arras et al.](https://arxiv.org/abs/2502.15886)

For more in-depth explanations, you can always refer to the [Documentation](https://lxt.readthedocs.io/).
We hope you find these examples helpful for both understanding the theory and applying LRP in practice!

Tested with `transformers==4.48.3`, `torch==2.6.0`, `python==3.11`.