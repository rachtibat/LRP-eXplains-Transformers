import operator
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxt.core import Composite
import lxt.functional as lf
import lxt.modules as lm
import lxt.rules as rules

#####################
### LRP Composite ###
#####################

# must be traced because of the functions, e.g. operator.add!
cp_lrp = Composite(
    {
        nn.MultiheadAttention: lm.MultiheadAttention_CP,
        # nn.Softmax: lm.SoftmaxDT,
        # order matters! lm.LinearInProjection is inside lm.MultiheadAttention_CP
        lm.LinearInProjection: rules.EpsilonRule,
        lm.LinearOutProjection: rules.EpsilonRule,
        nn.GELU: rules.IdentityRule,
        operator.add: lf.add2,
        operator.matmul: lf.matmul,
        nn.LayerNorm: lm.LayerNormEpsilon,
        # F.normalize: lf.normalize,
    }
)

# attnlrp = Composite(
#     {
#         nn.MultiheadAttention: lm.MultiheadAttention_AttnLRP,
#         nn.Softmax: lm.SoftmaxDT,
#         lm.LinearInProjection: rules.EpsilonRule,
#         lm.LinearOutProjection: rules.EpsilonRule,
#         nn.GELU: rules.IdentityRule,
#         operator.add: lf.add2,
#         operator.matmul: lf.matmul,
#         nn.LayerNorm: lm.LayerNormEpsilon,
#         # F.normalize: lf.normalize,
#     }
# )


def one_hot_max(output, targets):
    """
    Get the one-hot encoded max at the original indices in dim=1

    Args:
        output (torch.tensor): the output of the model
        targets (torch.tensor): the targets of the model

    Returns:
        torch.tensor: the one-hot encoded matrix multiplied by
                        the targets logit at the original indices in dim=1
    """
    values = output[torch.arange(output.shape[0]).unsqueeze(-1), targets.unsqueeze(-1)]
    eye_matrix = torch.eye(output.shape[-1]).to(output.device)
    return values * eye_matrix[targets]


def get_vit_b_16():
    from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # preprocess for ImageNet
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return model, preprocess


#####################
### Example Usage ###
#####################

if __name__ == "__main__":

    from PIL import Image
    from zennit.composites import LayerMapComposite
    import zennit.rules as z_rules
    from zennit.image import imgify

    device = "cuda"

    # Load a model
    model, preprocess = get_vit_b_16()
    model.eval()
    model = model.to(device)

    # Load an image and tokenize a text
    image = (
        #preprocess(Image.open("docs/source/_static/cat_dog.jpg"))
        preprocess(
            Image.open("docs/source/_static/cute_dog_cat_high.png").convert("RGB")
        )
        .unsqueeze(0)
        .to(device)
    )

    # trace the model with a dummy input
    # verbose=True prints all functions/layers found and replaced by LXT
    # you will see at the last entry that e.g. tensor.exp() is not supported by LXT. This is not a problem in our case,
    # because this function is not used in the backward pass and therefore does not need to be replaced.
    # (look into the open_clip.transformer module code!)
    x = torch.randn(1, 3, 224, 224, device=device)
    traced = cp_lrp.register(model, dummy_inputs={"x": x}, verbose=True)
    #traced = attnlrp.register(model, dummy_inputs={"x": x}, verbose=True)

    # for Vision Transformer, we must perform a grid search for the best gamma hyperparameters
    # in general, it is enough to concentrate on the Conv2d and MLP layers
    # for simplicity we just use a few values that can be evaluated by hand & looking at the heatmaps
    heatmaps = []
    for conv_gamma, lin_gamma in itertools.product([0.1, 0.25, 0.5], [0.01, 0.05, 0.1]):

        print("Gamma Conv2d:", conv_gamma, "Gamma Linear:", lin_gamma)

        # we define rules for the Conv2d and Linear layers using 'Zennit'
        zennit_comp = LayerMapComposite(
            [
                (nn.Conv2d, z_rules.Gamma(conv_gamma)),
                (nn.Linear, z_rules.Gamma(lin_gamma)),
            ]
        )

        # register composite
        zennit_comp.register(traced)

        # forward & backward pass
        y = traced(image.requires_grad_(True))
        # logits = lf.matmul(y[0], y[1].transpose(0, 1))
        max_logits, max_indices = torch.max(y[0], dim=-1)

        # explain the dog class ("a dog")
        image.grad = None
        # logits[0, 1].backward()
        # max_logits.backward(max_logits)

        # replace it with torch autograd
        (relevance,) = torch.autograd.grad(
            outputs=y,
            inputs=image,
            grad_outputs=one_hot_max(y, max_indices).to(device),
            retain_graph=False,
            create_graph=False,
        )
        print(relevance)
        # normalize the heatmap
        # heatmap = image.grad[0].sum(0)
        heatmap = relevance[0].sum(0)
        heatmap = heatmap / abs(heatmap).max()
        heatmaps.append(heatmap.cpu().numpy())

        # zennit composites can be removed so that we can register a new one!
        zennit_comp.remove()

    # save the heatmaps as a grid
    imgify(heatmaps, vmin=-1, vmax=1, grid=(3, 3)).save("heatmap_corgi_cp.png")
