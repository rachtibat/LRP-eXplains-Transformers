.. _under_the_hood:
How does LXT work under the hood?
=====

We have implemented `custom PyTorch autograd Functions <https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html>`_ for commonly used operations in transformers. 
These functions behave identically in the forward pass, but compute LRP attributions in the backward pass. 

Custom Autograd Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

For instance, the :math:`\varepsilon`-LRP rule for linear layers :math:`z = W x + b` is defined as

.. math::

   R^{l-1} = x \odot W^T \cdot \frac{R^l}{z + \varepsilon}


In ``lxt.functional``, we define a custom Autograd Function that applies the standard ``torch.nn.functional.linear`` function in the forward pass, but our LRP rule in the backward pass:

.. code-block:: python

   class linear_epsilon_fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weight, bias=None, epsilon=1e-6):
        
        # torch linear forward pass
        outputs = torch.nn.functional.linear(inputs, weight, bias)

        # save variables for backward pass
        ctx.save_for_backward(inputs, weight, outputs)
        ctx.epsilon = epsilon
    
        return outputs

    @staticmethod
    def backward(ctx, *out_relevance):

        inputs, weight, outputs = ctx.saved_tensors

        # apply epsilon-LRP equation
        relevance_norm = out_relevance[0] / (outputs + ctx.epsilon)
        relevance = torch.matmul(relevance_norm, weight).mul_(inputs)
        
        return (relevance, None, None, None)



Likewise, we also define more abstract "super-functions" that compute LRP rules using PyTorch's vector-Jacobian products for arbitrary modules.
These super-functions wrap modules and hence we do not need to replace each operation!
For instance, you can implement a super-function to compute :math:`\varepsilon`-LRP for any ``torch.nn.Module`` writing:

.. code-block:: python

    def my_super_function(module, inputs, out_relevance, epsilon=1e-6):

        outputs = module(inputs)

        relevance_norm = out_relevance / (outputs + epsilon)

        # computes vector-jacobian product
        grads = torch.autograd.grad(outputs, inputs, relevance_norm)

        relevance = grads * inputs
        return relevance


Very simple, isn't? (The module should compute a linear operation, otherwise the vector-Jacobian product will not result in the 
:math:`\varepsilon`-LRP rule, see Equation 8 in our paper `AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers <https://arxiv.org/abs/2402.05602>`_.).


How was the Quickstart demo created?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

So that LXT works properly, you have to replace **all** operations where the gradient is not equal to a relevance propagation rule.
For instance, in many projects you will find a line of code adding two tensors, such as ``hidden_states = hidden_states + residual``.

With LXT, we must replace this line of code with ``hidden_states = lxt.functional.add2(hidden_states, residual)``. 
For the Quickstart demos, we edited the huggingface source-code and replaced all operations that needed to be changed. 

In addition, LXT provides the capability to dynamically modify a portion of the source code. This functionality can be achieved by utilizing the ``Composite`` class, which is described in detail in :ref:`on_the_fly`. 
In order to *easily* compare AttnLRP with Conservative Propagation (CP)-LRP in our paper, we added custom ``nn.Modules`` wrapper around key-components of the model, where AttnLRP and CP-LRP differ such as ``nn.Softmax``.
Then, we applied different LRP rules to ``nn.Modules`` using different ``Composites``.

Hence, you will find in each model that we provide a ``attnlrp`` and ``cp_lrp`` composite.