.. _under_the_hood_efficient:
How does the Gradient*Input Framework work?
============================================

Since LRP is a backpropagation-based method, it is possible to compute attributions in a single backward pass.
However, computing the standard LRP rules as defined in the original paper requires additional operations that are not part of the standard backpropagation algorithm in PyTorch.
This inflicts additional computational costs, which can be reduced by reforumlating the LRP rules in such a way that reduce the number of operations.

This elegant way of implementing AttnLRP was introduced in `Arras, et al. “Close Look at Decomposition-based XAI-Methods for Transformer Language Models.” arXiv preprint, 2025. <https://arxiv.org/abs/2502.15886>`_


Standard MLP Block in LLMs
~~~~~~~~~~~~~~~~~~~~~~~~~

Let's take a look at a standard MLP block. Usually it consists of two linear layers with a non-linear activation function in between.
The forward pass of the block can be written as:

.. math::

    x_2 = W_1 \cdot x_1

    x_3 = \text{SiLU}(x_2)

    z = W_2 \cdot x_3

where :math:`x_1` is the input, :math:`W_1` and :math:`W_2` are the weights, and :math:`z` is the output.

To explain this block, we apply the :math:`\varepsilon`-LRP on linear layers and the identity-rule on the activation function.
As a reminder, the :math:`\varepsilon`-LRP rule for a linear layer :math:`z = W x + b` is defined as

.. math::

   R^{l-1} = x \odot W^T \cdot \frac{R^l}{z + \varepsilon}


Now, we apply it backwards in sequence to the MLP block:

.. math::

    R^{l-1} = x_3 \odot W_2^T \cdot \frac{R^l}{z + \varepsilon}

    R^{l-2} = R^{l-1} \quad \text{(identity rule)}

    R^{l-3} = x_1 \odot W_1^T \cdot \frac{R^{l-2}}{x_2 + \varepsilon}

Now, let's write it in one line beginning from the last equation:

.. math::

    R^{l-3} = x_1 \odot W_1^T \cdot \frac{\text{SiLU}(x_2)}{x_2 + \varepsilon} \odot W_2^T \cdot \frac{R^l}{z + \varepsilon}

We notice, that :math:`W_1^T` and :math:`W_2^T` are the Jacobians of the linear layers that are returned by PyTorch's backward pass.
The only anomaly is the term :math:`\frac{\text{SiLU}(x_2)}{x_2 + \varepsilon}`.
Assumed, we have a special function that returns a Jacobian :math:`J_2` equal to :math:`\frac{\text{SiLU}(x_2)}{x_2 + \varepsilon}`, then we can rewrite this as a chain of Jacobian-vector products:

.. math::

    R^{l-3} = x_1 \odot J_1 \cdot J_2 \cdot J_3 \cdot \frac{R^l}{z + \varepsilon}

Assumed, we want to explain only one dimension of :math:`z`, i.e. :math:`z_i`, we set the relevance :math:`R_i^l` to :math:`z_i` and all other elements to zero.
Then, it reduces to a one-hot vector that is multiplied with the Jacobians, which is equivalent to a simple gradient computation.

At the end, we have a chain of Jacobian-vector products that only needs to be multiplied with the input :math:`x_1` to get the attributions.

.. math::

    R^{l-3} = x_1 \odot \text{Gradient}

This means, we only need to make sure that the gradient at the SiLU layer is computed correctly.
We can do that by e.g. defining a custom Autograd function in PyTorch that computes an identical SiLU forward pass but this modified gradient in the backward pass.

.. code-block:: python

    class SiLUWithModifiedGradient(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x_2, epsilon=1e-10):

            x_3 = nn.SiLU()(x_2)
            ctx.save_for_backward(x_3/(x_2 + epsilon))
            return output

        @staticmethod
        def backward(ctx, *out_relevance):

            gradient = ctx.saved_tensors[0] * out_relevance[0]
            return gradient, None

Now, we compute ``z[i].backward()`` in PyTorch, and multiply the input with the gradient ``x_1 * x_1.grad`` to get the attributions.
The same methodolgy can be applied to the attention and normalization layers in LLMs.

The beauty of this approach is, that PyTorch already returns the correct Jacobian for all linear operation, e.g. addition, and we only need to modify the gradient at the SiLU layer,
attention and normalization layers. Three modifications that are easy to implement and that reduce the computational costs significantly.
