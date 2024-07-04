import operator
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxt.core import Composite
import lxt.functional as lf
import lxt.modules as lm
import lxt.rules as rules

import open_clip
import open_clip.transformer


#####################
### LRP Composite ###
#####################

# must be traced because of the functions, e.g. operator.add!
attnlrp = Composite({
        nn.MultiheadAttention: lm.MultiheadAttention_CP,
        # order matters! lm.LinearInProjection is inside lm.MultiheadAttention_CP
        lm.LinearInProjection: rules.EpsilonRule,
        lm.LinearOutProjection: rules.EpsilonRule,
        open_clip.transformer.LayerNorm: lm.LayerNormEpsilon,
        nn.GELU: rules.IdentityRule,
        
        operator.add: lf.add2,
        operator.matmul: lf.matmul,
        F.normalize: lf.normalize,
    })

#####################
### Example Usage ###
#####################

if __name__ == "__main__":

    import open_clip
    from PIL import Image
    from zennit.composites import LayerMapComposite
    import zennit.rules as z_rules
    from zennit.image import imgify

    device = 'cuda'

    # Load the model and the tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
    model.eval()
    model = model.to(device)

    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    # Load an image and tokenize a text
    text = tokenizer(['a beautiful LRP heatmap', 'a dog', 'a cat']).to(device)
    image = preprocess(Image.open('docs/source/_static/cat_dog.jpg')).unsqueeze(0).to(device)

    # trace the model with a dummy input
    # verbose=True prints all functions/layers found and replaced by LXT
    # you will see at the last entry that e.g. tensor.exp() is not supported by LXT. This is not a problem in our case,
    # because this function is not used in the backward pass and therefore does not need to be replaced.
    # (look into the open_clip.transformer module code!)
    x = torch.randn(1, 3, 224, 224, device=device)
    traced = attnlrp.register(model, dummy_inputs={'image': x, 'text': text}, verbose=True)

    # for Vision Transformer, we must perform a grid search for the best gamma hyperparameters
    # in general, it is enough to concentrate on the Conv2d and MLP layers
    # for simplicity we just use a few values that can be evaluated by hand & looking at the heatmaps
    heatmaps = []
    for conv_gamma, lin_gamma in itertools.product([0.1, 0.5, 100], [0, 0.01, 0.05, 0.1, 1]):

        print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)

        # we define rules for the Conv2d and Linear layers using 'Zennit'
        zennit_comp = LayerMapComposite([
                (nn.Conv2d, z_rules.Gamma(conv_gamma)),
                (nn.Linear, z_rules.Gamma(lin_gamma)),
            ])

        # register composite
        zennit_comp.register(traced)

        # forward & backward pass
        y = traced(image.requires_grad_(True), text)
        logits = lf.matmul(y[0], y[1].transpose(0, 1))

        # explain the dog class ("a dog")
        image.grad = None
        logits[0, 1].backward()

        # normalize the heatmap
        heatmap = image.grad[0].sum(0)
        heatmap = heatmap / abs(heatmap).max()
        heatmaps.append(heatmap.cpu().numpy())

        # zennit composites can be removed so that we can register a new one!
        zennit_comp.remove()

    # save the heatmaps as a grid
    imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 5)).save('heatmap.png')

