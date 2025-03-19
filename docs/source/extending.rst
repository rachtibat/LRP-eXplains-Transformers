.. _extending:

Supported Models & Extending LXT
=========================================
Moving forward, we will exclusively maintain the efficient variant of the LXT library, which will allow us to deliver more consistent performance improvements and reduce maintenance overhead.

List of supported Models
~~~~~~~~~~~~~~~~~~~~~~~~~
This is a not exhaustive list of models that are supported in the efficient LXT variant due to active development.
For a complete list of models, please look into ``lxt.efficient.models``.

- LLaMA
- Qwen
- BERT
- GPT2
- torchvision ViT

Coming soon:
- Gemma


Extending LXT
~~~~~~~~~~~~~~
Contributions to LXT are highly welcome! The explicit variant will not be maintained anymore, but we will continue to support the efficient variant.
Hence, please do not add new models to the explicit variant.

To add a new model, create a new file in the ``lxt/efficient/models`` directory and implement the patching logic for the model.
At the end, please add your model to the ``DEFAULT_MAP`` in the ``__init__.py`` file in the ``lxt/efficient/models`` directory.

You could either edit the original model source code directly or use the monkey patching functionality to make it more future-proof,
because Huggingface is regularly updating their models and will break dependencies. However, manual editing is easier.

**Manual Editing:**

In general, you only need to change three things in LLMs. You can find an example in the ``lxt/efficient/models/bert.py`` file, where all 
changes are commented with ``### <------------------------------------------- LXT``.

1. Apply the identity rule on element-wise non-linearities, such as GELU

.. code-block:: python

    from lxt.efficient.rules import identity_rule_implicit

    # some MLP block
    def forward(self, x):
        x = self.fc1(x)
        x = identity_rule_implicit(self.activation, x) # Apply the identity rule
        x = self.fc2(x)
        return x

2. If required, apply also uniform rule on element-wise multiplication in gated MLPs.

.. code-block:: python

    from lxt.efficient.rules import divide_gradient

    # some MLP block
    def forward(self, x):

        gate_out = self.gate_proj(x)
        gate_out = identity_rule_implicit(self.act_fn, gate_out) # Apply the identity rule

        weighted = gate_out * self.up_proj(x)
        weighted = divide_gradient(weighted, 2) # Apply the uniform rule
        return self.down_proj(weighted)


3. Apply the uniform rule on matrix multiplications.

.. code-block:: python

    from lxt.efficient.rules import divide_gradient

    # some attention function
    def forward(self, key, value, query):
        # ....
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = divide_gradient(attention_scores, 2) # Apply the uniform rule
        # ....
        attn_weights = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = divide_gradient(attn_output, 2) # Apply the uniform rule
        # ....

For sdpa or flash attention, you can simply apply the uniform rule later at the inputs/outputs

.. code-block:: python

    from lxt.efficient.rules import divide_gradient
    
    # some attention function
    def forward(self, key, value, query):
        # ....
        query = divide_gradient(query, 2) # Apply the uniform rule for query @ key multiplication
        key = divide_gradient(key, 2) # Apply the uniform rule for query @ key multiplication

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0, # No dropout if model is in train() mode
            is_causal=is_causal,
        )

        attn_output = divide_gradient(attn_output, 2) # Apply the uniform rule for softmax @ value multiplication
        # ....


4. Apply the identity rule on the normalization operation inside the RMSNorm layer.
To do this, we can simply stop the gradient flow through the variance computation.

.. code-block:: python

    from lxt.efficient.rules import stop_gradient

    # some RMSNorm layer
    def forward(self, x):
        # ....
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / stop_gradient(torch.sqrt(var + self.eps)) # Stop the gradient flow
        # ....

**Patching:**

If the model follows the standard huggingface model structure, you can use the ``lxt.efficient.patches`` functionality.
As reference you can look into the LLaMA model in the ``lxt/efficient/models/llama.py`` file. Then, you only change three things

1. Patch the attention functions (``eager_attention_forward``, ``ALL_ATTENTION_FUNCTIONS``) with ``lxt.efficient.patches.patch_attention``, which will automatically apply the uniform rule to various ``torch.matmul`` operations.
2. Patch the forward pass in the RMS-Norm layers which stops the gradient flow through the variance computation.
3. Patch the forward pass in the MLP Block which applies the identity rule to the non-linear activation function & apply the uniform rule to a gated multiplication, if availble.

