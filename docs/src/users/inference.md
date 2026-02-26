# Inference

Briefly, you need to:

1. Download a checkpoint.
2. Get the code.
3. Load the checkpoint.
4. Get activations.

Details are below.

## Download a Checkpoint

First, download an SAE checkpoint from the [Huggingface collection](https://huggingface.co/collections/osunlp/sae-v-67ab8c4fdf179d117db28195).

### Single-checkpoint repos

Some repos (CLIP, BioCLIP, DINOv2) contain a single `sae.pt` at the root. For instance, the SAE trained on OpenAI's CLIP ViT-B/16 with ImageNet-1K activations is [here](https://huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K).

You can use `wget` if you want:

```sh
wget https://huggingface.co/osunlp/SAE_CLIP_24K_ViT-B-16_IN1K/resolve/main/sae.pt
```

### Multi-checkpoint repos

The DINOv3 repos contain multiple checkpoints organized by layer and sparsity level. Each repo has a `manifest.jsonl` with metadata (layer, L0, MSE) for every checkpoint, so you can pick the right one programmatically.

Download a specific checkpoint:

```python
from huggingface_hub import hf_hub_download

# Pick a specific layer and run ID from the repo's README or manifest.jsonl
path = hf_hub_download("osunlp/SAE_DINOv3_ViT-L-16_IN1K", "layer_23/lnleoyf6/sae.pt")
```

Download all checkpoints in a repo:

```python
from huggingface_hub import snapshot_download

snapshot_download("osunlp/SAE_DINOv3_ViT-L-16_IN1K")
```

Available DINOv3 repos:

- [osunlp/SAE_DINOv3_ViT-S-16_IN1K](https://huggingface.co/osunlp/SAE_DINOv3_ViT-S-16_IN1K) (layers 6-11)
- [osunlp/SAE_DINOv3_ViT-B-16_IN1K](https://huggingface.co/osunlp/SAE_DINOv3_ViT-B-16_IN1K) (layers 6-11)
- [osunlp/SAE_DINOv3_ViT-L-16_IN1K](https://huggingface.co/osunlp/SAE_DINOv3_ViT-L-16_IN1K) (layers 13-23)
- [osunlp/SAE_DINOv3_TopK_ViT-L-16_IN1K](https://huggingface.co/osunlp/SAE_DINOv3_TopK_ViT-L-16_IN1K) (layers 13-23)

## Get the Code

The easiest way to do this is to clone the code:

```
git clone https://github.com/OSU-NLP-Group/saev
```

You can also install the package from git if you use uv (not sure about pip or cuda):

```sh
uv add git+https://github.com/OSU-NLP-Group/saev
```

Or clone it and install it as an editable with pip, lik `pip install -e .` in your virtual environment.

Then you can do things like `from saev import ...`.

!!! note

    If you struggle to get `saev` installed, open an issue on [GitHub](https://github.com/OSU-NLP-Group/saev) and I will figure out how to make it easier.

## Load the Checkpoint

```py
import saev.nn

sae = saev.nn.load("PATH_TO_YOUR_SAE_CKPT.pt")
```

Now you have a pretrained SAE.

## Get Activations

This is the hardest part.
We need to:

1. Pass an image into a ViT
2. Record the dense ViT activations at the same layer that the SAE was trained on.
3. Pass the activations into the SAE to get sparse activations.
4. Do something interesting with the sparse SAE activations.

There are examples of this in the demo code: for [classification](https://huggingface.co/spaces/samuelstevens/saev-image-classification/blob/main/app.py#L318) and [semantic segmentation](https://huggingface.co/spaces/samuelstevens/saev-semantic-segmentation/blob/main/app.py#L222).
If the permalinks change, you are looking for the `get_sae_latents()` functions in both files.

Below is example code to do it using the `saev` package.

```py
import saev.nn
import saev.data.models
import saev.data.shards

vit_cls = saev.data.models.load_model_cls("clip")
vit = vit_cls("ViT-B-16/openai").to(device)
vit = saev.data.shards.RecordedTransformer(vit, 196, True, [10])

img_tr, _ = vit_cls.make_transforms("ViT-B-16/openai", 196)
img = Image.open("example.jpg")

x = img_transform(img)
# Add a batch dimension
x = x[None, ...]
_, vit_acts = recorded_vit(x)
# Select the only layer in the batch and ignore the CLS token.
vit_acts = vit_acts[:, 0, 1:, :]

x_hat, f_x, loss = sae(vit_acts)
```

Now you have the reconstructed x (`x_hat`) and the sparse representation of all patches in the image (`f_x`).

You might select the dimensions with maximal values for each patch and see what other images are maximimally activating.
