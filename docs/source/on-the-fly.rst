.. _on_the_fly:
On-the-fly Modifications
=====

To understand how LXT works under the hood, you can read :ref:`under_the_hood`.

There are two ways to apply LXT to your model:

#. either by modifying your source code by using drop-in-replacements for your functions and modules
#. or modifying your source code on-the-fly with the Composite class

Here, we discuss point 2.


Rules
~~~~~


In contrast to the ``lxt.functional`` and ``lxt.modules`` drop-in-replacements, we also provide more abstract "super-functions" that compute LRP rules using PyTorch's vector-Jacobian products for arbitrary modules.
These super-functions wrap modules and hence we do not need to replace each operation!

These rules must be used inside a ``lxt.core.Composite``, that we will show below.
For now, here you see a non-exhaustive list of available rules:


.. list-table:: Rules (super-functions)
    :widths: 25 25 50
    :header-rows: 1

    * - Name
      - LXT
      - Description
    * - :math:`\varepsilon`-LRP
      - lxt.rules.EpsilonRule
      - standard :math:`\varepsilon`-LRP
    * - Uniform Rule
      - lxt.rules.UniformRule
      - distributes relevance uniformely to all input arguments
    * - Uniform and :math:`\varepsilon`-LRP
      - lxt.rules.UniformEpsilonRule
      - sequential application of the :math:`\varepsilon`-LRP and uniform rule
    * - Identity Rule
      - lxt.rules.IdentityRule
      - passes the relevance without modification through to the input variables
    * - Stop Relevance Flow
      - lxt.rules.StopRelevanceRule
      - stops the relevance flow by setting input relevances to None (zero)
    * - Deep Taylor Decomposition
      - lxt.rules.TaylorDecompositionRule
      - standard Deep Taylor Decomposition with or without bias



The Composite
~~~~~~~~~~~~~

To attach rules and replace modules in your model, we provide the ``lxt.core.Composite`` class.
This class replaces the attributes of your model with LXT variants.

The Composite takes as argument a dictionary, where the keys represent ``nn.Module`` types and the values are either ``lxt.rules`` or ``lxt.modules``.

Let's say we have a simple Sequential model containing a linear and a root-mean-square normalization layer.
Then, we'd like to apply the :math:`\varepsilon`-LRP on the linear layer and the identity rule (passing the relevence through) to the normalization layer.

.. code-block:: python

    from lxt.core import Composite
    import lxt.rules as rules

    model = nn.Sequential(
        nn.Linear(10, 10),
        RootMeanSquareNorm(),
    )

    lrp = Composite({
        nn.Linear: rules.EpsilonRule,
        RootMeanSquareNorm: rules.IdentityRule,
    }, verbose=True)
    
    # wrap modules in LXT rules and show the progress
    lrp.register(model, verbose=True)

    # print model to see the modification
    print(model)

    y = model(x.requires_grad_())

    y.max().backward()

    relevance = x.grad


That's it! If you look at the print statement in your console, you will see that the modules are indeed wrapped with the LXT rules.

.. raw:: html

    <embed src="_static/terminal.png" height="200">

You could also supply ``lxt.modules`` instead of ``lxt.rules``, such as 

.. code-block:: python

  lrp = Composite({
          nn.Linear: lxt.modules.LinearEpsilon,
          RootMeanSquareNorm: lxt.modules.RMSNormIdentity,
      })

To revert the modification, simply write

.. code-block:: python

  lrp.remove()

  # print model to see the modification
  print(model)

and you should see in the terminal that the rules are removed. 
(This only works if the model was not symbolically traced as explained later. There will be a warning message if something went wrong.)


torch.fx Graph Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
  torch.fx is not compatible with gradient checkpointing and some models are not symbolically tracable!


So that LXT works properly, you have to replace **all** operations where the gradient is not equal to a relevance propagation rule.
For instance, in many projects you will find a line of code adding two tensors, such as ``hidden_states = hidden_states + residual``.

With LXT, we must replace this line of code with ``hidden_states = lxt.functional.add2(hidden_states, residual)``. 
However, since replacing all lines might be tedious, we exploited ``torch.fx`` to replace these operations for us automatically!

To use ``torch.fx``, you must supply a dummy input

.. code-block:: python

  import torch
  import operator
  import lxt

  class SimpleModel(torch.nn.Module):

      def __init__(self):
          super().__init__()

          self.layer1 = torch.nn.Linear(10, 20, True)
          self.layer2 = torch.nn.Linear(10, 20, True)

      def forward(self, x):

          y1 = self.layer1(x)
          y2 = self.layer2(x)

          y1 = torch.nn.functional.softmax(y1, -1)

          return y1 + y2

      model = SimpleModel()

      lrp = Composite({
          nn.Linear: lxt.rules.EpsilonRule,
          operator.add: lxt.functional.add2,
          torch.nn.functional.softmax: lxt.functional.softmax,
      })

      x = torch.randn(1, 32, 10, requires_grad=True)
      model = lrp.register(model, dummy_inputs={'x': x}, verbose=True)

      print(model)

You can not remove composites from traced models i.e. ``lrp.remove()`` will not work!
(You will see a warning message)