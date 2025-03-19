.. _drop_in_replacement:
LXT Drop-in replacements
========================

To understand how LXT works under the hood, you can read :ref:`under_the_hood`.

There are two ways to apply LXT to your model:

#. either by modifying your source code by using drop-in-replacements for your functions and modules
#. or modifying your source code on-the-fly with the Composite class

Here, we discuss point 1.


Functionals
~~~~~~~~~~~~

We have implemented `custom PyTorch autograd Functions <https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_ for commonly used operations in transformers. 
These functions behave identically in the forward pass, but compute LRP attributions in the backward pass. You can use them as drop-in-replacement in your code. 

For example, to compute the :math:`\varepsilon`-LRP rule for a linear function :math:`z = W x + b`, you can simply replace ``torch.nn.functional.linear`` with

.. code-block:: python

    import lxt.explicit.functional as lf

    y = lf.linear_epsilon(x, W, b)

    # initialize relevance with y itself
    y.backward(y)

    relevance = x.grad

    # or for instance explain max output only
    y = lf.linear_epsilon(x, W, b)
    y.max().backward()

    relevance = x.grad


Here is a non-exhaustive table of functionals that we provide. The Equations and Propositions are described in our paper
`AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers <https://arxiv.org/abs/2402.05602>`_.

.. list-table:: Functions
    :widths: 25 25 50
    :header-rows: 1

    * - torch
      - LXT
      - Description (Reference to paper)
    * - torch.nn.functional.linear
      - lxt.functional.linear_epsilon
      - standard :math:`\varepsilon`-LRP for a linear layer according to the Equation 8
    * - torch.add
      - lxt.functional.add2
      - :math:`\varepsilon`-LRP for the addition of two tensors according to the Equation 8
    * - torch.nn.functional.softmax
      - lxt.functional.softmax
      - Deep Taylor Decomposition with bias for Softmax according to Proposition 3.1
    * - torch.mul
      - lxt.functional.mul2
      - uniform rule for elementwise multiplication (along all dimensions) of two tensors according to Proposition 3.2. If one input is a constant, the identity rule is applied.
    * - torch.matmul
      - lxt.functional.matmul
      - sequential application of the :math:`\varepsilon`-LRP and uniform rule for matrix multiplication according to Proposition 3.3
    * - 
      - lxt.functional.rms_norm_identity
      - computes the root-mean-squared normalization in forward pass and the identity rule in backward according to Proposition 3.4




Modules
~~~~~~~~~~

We also wrapped some functions into ``nn.Modules`` so that you can use them as drop-in-replacement in your code. 

For example, to compute the :math:`\varepsilon`-LRP rule for a linear layer :math:`z = W x + b`, you can simply replace ``torch.nn.Linear`` with
   
.. code-block:: python

    import lxt.explicit.modules as lm

    layer = lm.LinearEpsilon(10, 20)

    y = layer(x)

    # e.g. initialize relevance with y itself
    y.backward(y)

    relevance = x.grad


Here is a non-exhaustive table of modules that we provide. The Equations and Propositions are described in our paper
`AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers <https://arxiv.org/abs/2402.05602>`_.

.. list-table:: Modules
    :widths: 25 25 50
    :header-rows: 1

    * - torch.nn
      - LXT
      - Description (Reference to paper)
    * - Linear
      - lxt.modules.LinearEpsilon
      - standard :math:`\varepsilon`-LRP for a linear layer according to the Equation 8
    * - Softmax
      - lxt.modules.SoftmaxDT
      - Deep Taylor Decomposition with bias for Softmax according to Proposition 3.1
    * - 
      - lxt.modules.RMSNormIdentity
      - computes the root-mean-squared normalization in forward pass and the identity rule in backward according to Proposition 3.4