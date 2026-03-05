import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import urllib.request
    from collections.abc import Callable

    import marimo as mo
    import matplotlib.pyplot as plt
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    import saev.data.models
    import saev.data.shards
    import saev.nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return (
        Callable,
        Image,
        device,
        hf_hub_download,
        math,
        mo,
        plt,
        saev,
        torch,
        urllib,
    )


@app.cell(hide_code=True)
def _(device, mo):
    mo.md(f"""
    Using device `{device}`
    """)
    return


@app.cell
def _(Callable, Image, hf_hub_download, plt, saev, torch):
    def load_model_and_sae(
        family: str,
        ckpt: str,
        sae_hf_repo: str,
        layer: int,
        n_content_tokens: int,
        device: str,
    ) -> tuple[saev.data.shards.RecordedTransformer, saev.nn.SparseAutoencoder, Callable]:
        """Load a ViT and its corresponding SAE checkpoint.

        Returns (recorded_vit, sae, img_transform).
        """
        vit_cls = saev.data.models.load_model_cls(family)

        # Get transforms before constructing the model to avoid loading weights
        # twice for CLIP-family models (make_transforms calls
        # create_model_from_pretrained internally).
        img_tr, _ = vit_cls.make_transforms(ckpt, n_content_tokens)

        vit = vit_cls(ckpt).to(device)
        vit.eval()
        recorded_vit = saev.data.shards.RecordedTransformer(
            vit, n_content_tokens, True, [layer]
        )

        sae_fpath = hf_hub_download(sae_hf_repo, "sae.pt")
        sae = saev.nn.load(sae_fpath, device=device)
        sae.eval()

        return recorded_vit, sae, img_tr


    def extract_features(
        recorded_vit: saev.data.shards.RecordedTransformer,
        sae: saev.nn.SparseAutoencoder,
        img: Image.Image,
        img_tr: Callable,
        device: str,
    ):
        """Run an image through the ViT and SAE.

        Returns (patch_acts, sae_out) where patch_acts has shape
        [n_content_tokens, d_model] and sae_out.f_x has shape
        [n_content_tokens, d_sae].
        """
        x = img_tr(img)
        x = x[None, ...].to(device)

        with torch.no_grad():
            _, vit_acts = recorded_vit(x)

        # vit_acts: [batch, n_layers, tokens_per_example, d_model]
        # Select layer 0, strip CLS token (index 0).
        patch_acts = vit_acts[0, 0, 1:, :]

        with torch.no_grad():
            sae_out = sae(patch_acts)

        return patch_acts, sae_out


    def select_top_latents(f_x, k=5):
        """Pick top-k latents by spatial variance of activation.

        Spatial variance highlights latents that activate in localized regions
        rather than uniformly or in a single spike.
        """
        # f_x: [n_content_tokens, d_sae]
        variance = f_x.var(dim=0)
        _, top_i = variance.topk(k)
        return top_i


    def plot_latent_heatmaps(img_224, f_x, latent_i, grid_h, grid_w):
        """Plot heatmap overlays on the 224x224 crop for each latent."""
        n_latents = len(latent_i)
        fig, axes = plt.subplots(1, n_latents, figsize=(4 * n_latents, 4))
        if n_latents == 1:
            axes = [axes]

        for ax, li in zip(axes, latent_i):
            heatmap = f_x[:, li].reshape(grid_h, grid_w).float().cpu().numpy()
            vmin, vmax = heatmap.min(), heatmap.max()
            if vmax > vmin:
                heatmap = (heatmap - vmin) / (vmax - vmin)

            ax.imshow(img_224)
            ax.imshow(
                heatmap,
                alpha=0.5,
                cmap="hot",
                interpolation="bilinear",
                extent=(0, 224, 224, 0),
            )
            ax.set_title(f"Latent {li.item()}")
            ax.axis("off")

        fig.tight_layout()
        return fig

    return (
        extract_features,
        load_model_and_sae,
        plot_latent_heatmaps,
        select_top_latents,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Example image**
    """)
    return


@app.cell
def _(Image, mo, urllib):
    with urllib.request.urlopen(
        urllib.request.Request(
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg",
            headers={"User-Agent": "saev-demo/1.0"},
        )
    ) as _resp:
        img = Image.open(_resp).convert("RGB")

    mo.image(img, width=400)
    return (img,)


@app.cell
def _(device, load_model_and_sae):
    dino_vit, dino_sae, dino_tr = load_model_and_sae(
        "dinov2",
        "dinov2_vitb14",
        "osunlp/SAE_DINOv2_24K_ViT-B-14_IN1K",
        10,
        256,
        device,
    )
    bio_vit, bio_sae, bio_tr = load_model_and_sae(
        "clip",
        "hf-hub:imageomics/bioclip",
        "osunlp/SAE_BioCLIP_24K_ViT-B-16_iNat21",
        10,
        196,
        device,
    )
    return bio_sae, bio_tr, bio_vit, dino_sae, dino_tr, dino_vit


@app.cell
def _(bio_vit, device, dino_vit, mo):
    mo.md(f"""
    **Models loaded.**

    | Model | patch_size | content tokens | device |
    |-------|-----------|---------------|--------|
    | DINOv2 ViT-B/14 | {dino_vit.model.patch_size} | 256 | `{device}` |
    | BioCLIP ViT-B/16 | {bio_vit.model.patch_size} | 196 | `{device}` |
    """)
    return


@app.cell
def _(
    bio_sae,
    bio_tr,
    bio_vit,
    device,
    dino_sae,
    dino_tr,
    dino_vit,
    extract_features,
    img,
):
    dino_patch_acts, dino_out = extract_features(dino_vit, dino_sae, img, dino_tr, device)
    bio_patch_acts, bio_out = extract_features(bio_vit, bio_sae, img, bio_tr, device)
    return bio_out, dino_out


@app.cell
def _(bio_out, dino_out, mo):
    def _sparsity_stats(name, f_x):
        n_nonzero = (f_x > 0).sum().item()
        total = f_x.numel()
        mass = f_x.sum().item()
        return f"**{name}:** f_x shape `{tuple(f_x.shape)}`, nonzero {n_nonzero}/{total} ({100 * n_nonzero / total:.1f}%), activation mass {mass:.1f}"


    mo.md(
        "\n\n".join(
            [
                "**Feature extraction complete.**",
                _sparsity_stats("DINOv2", dino_out.f_x),
                _sparsity_stats("BioCLIP", bio_out.f_x),
            ]
        )
    )
    return


@app.cell
def _(
    bio_out,
    dino_out,
    img,
    math,
    mo,
    plot_latent_heatmaps,
    select_top_latents,
):
    # Get 224x224 crops for overlay. DINOv2 and BioCLIP both produce 224px crops
    # via different transform pipelines, but we need the PIL image for display.
    from torchvision.transforms import v2 as _v2

    _crop = _v2.Compose([_v2.Resize(256), _v2.CenterCrop(224)])
    img_224 = _crop(img)

    dino_grid = int(math.isqrt(dino_out.f_x.shape[0]))
    bio_grid = int(math.isqrt(bio_out.f_x.shape[0]))

    dino_top = select_top_latents(dino_out.f_x, k=5)
    bio_top = select_top_latents(bio_out.f_x, k=5)

    dino_fig = plot_latent_heatmaps(img_224, dino_out.f_x, dino_top, dino_grid, dino_grid)
    dino_fig.suptitle("DINOv2 - Top 5 latents by spatial variance", y=1.02)

    bio_fig = plot_latent_heatmaps(img_224, bio_out.f_x, bio_top, bio_grid, bio_grid)
    bio_fig.suptitle("BioCLIP - Top 5 latents by spatial variance", y=1.02)

    mo.vstack([mo.as_html(dino_fig), mo.as_html(bio_fig)])
    return bio_grid, bio_top, dino_grid, dino_top, img_224


@app.cell
def _(
    bio_grid,
    bio_out,
    bio_top,
    dino_grid,
    dino_out,
    dino_top,
    img_224,
    mo,
    plt,
):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for row, (name, f_x, top_i, gh, gw) in enumerate(
        [
            ("DINOv2", dino_out.f_x, dino_top[:3], dino_grid, dino_grid),
            ("BioCLIP", bio_out.f_x, bio_top[:3], bio_grid, bio_grid),
        ]
    ):
        for col, li in enumerate(top_i):
            ax = axes[row, col]
            heatmap = f_x[:, li].reshape(gh, gw).float().cpu().numpy()
            vmin, vmax = heatmap.min(), heatmap.max()
            if vmax > vmin:
                heatmap = (heatmap - vmin) / (vmax - vmin)
            ax.imshow(img_224)
            ax.imshow(
                heatmap,
                alpha=0.5,
                cmap="hot",
                interpolation="bilinear",
                extent=(0, 224, 224, 0),
            )
            ax.set_title(f"{name} latent {li.item()} (max={f_x[:, li].max():.2f})")
            ax.axis("off")

    fig.suptitle("Side-by-side: Top 3 SAE latents per model", fontsize=14)
    fig.tight_layout()
    mo.as_html(fig)
    return


if __name__ == "__main__":
    app.run()
