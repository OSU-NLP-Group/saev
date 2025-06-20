import marimo

__generated_with = "0.9.32"
app = marimo.App(
    width="full",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    n_images_per_feature = 5
    n_features = 3

    pw_px, ph_px = (14, 14)

    colors = [
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
        "#ff7f00",
        "#a6cee3",
        "#b2df8a",
        "#fb9a99",
        "#fdbf6f",
    ]
    return colors, n_features, n_images_per_feature, ph_px, pw_px


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        This is a dashboard to explore how you can use sparse autoencoders to manipulate *semantic segmentations*. It presents a couple different ways to automatically propose features that you might want to manipulate.

        How do we propose features?

        1. Pick the 3 features that maximally activate on a target class's patches. This demonstrates that we can influence decisions for/against a class.
        3. Pick 3 random features. This demonstrates that arbitrary, unrelated directions don't really matter for downstream predictions (precision).

        For all of these features, show the `n_images_per_feature` highest activating examples from ImageNet-1K.
        """
    )
    return


@app.cell
def __(contrib, saev):
    head_ckpt_fpath = "/home/stevens.994/projects/saev/checkpoints/contrib/semseg/lr_0_001__wd_0_001/model_step8000.pt"
    head = contrib.semseg.training.load(head_ckpt_fpath)
    head.eval()

    sae_ckpt_fpath = "checkpoints/oebd6e6i/sae.pt"
    sae = saev.nn.load(sae_ckpt_fpath)
    sae.eval()
    return head, head_ckpt_fpath, sae, sae_ckpt_fpath


@app.cell
def __(einops, saev, v2):
    acts_cfg = saev.config.DataLoad(
        shard_root="/local/scratch/stevens.994/cache/saev/1864947033ca8b8a171a482644a948a6e6489e3249469373c78dfeeb0a75bcd4",
        scale_mean=True,
        scale_norm=True,
    )

    acts_dataset = saev.activations.Dataset(acts_cfg)

    imgs_cfg = saev.config.Ade20kDataset(
        root="/research/nfs_su_809/workspace/stevens.994/datasets/ade20k/"
    )

    to_array = v2.Compose([
        v2.Resize(256, interpolation=v2.InterpolationMode.NEAREST),
        v2.CenterCrop((224, 224)),
        v2.ToImage(),
        einops.layers.torch.Rearrange("channels width height -> width height channels"),
    ])

    imgs_dataset = saev.activations.Ade20k(
        imgs_cfg, img_transform=to_array, seg_transform=to_array
    )

    assert len(imgs_dataset) * acts_dataset.metadata.n_patches_per_img == len(
        acts_dataset
    )
    return acts_cfg, acts_dataset, imgs_cfg, imgs_dataset, to_array


@app.cell
def __(os, torch):
    sae_data_root = "/research/nfs_su_809/workspace/stevens.994/saev/features/oebd6e6i/sort_by_patch"

    top_img_i = torch.load(
        os.path.join(sae_data_root, "top_img_i.pt"),
        weights_only=True,
        map_location="cpu",
    )
    top_values = torch.load(
        os.path.join(sae_data_root, "top_values.pt"),
        weights_only=True,
        map_location="cpu",
    )
    sparsity = torch.load(
        os.path.join(sae_data_root, "sparsity.pt"),
        weights_only=True,
        map_location="cpu",
    )
    percentiles = torch.load(
        os.path.join(sae_data_root, "percentiles_p99.pt"),
        weights_only=True,
        map_location="cpu",
    )
    return percentiles, sae_data_root, sparsity, top_img_i, top_values


@app.cell
def __(mo):
    get_example_idx, set_example_idx = mo.state(1)
    return get_example_idx, set_example_idx


@app.cell
def __(imgs_dataset, mo, random, set_example_idx):
    random_example_btn = mo.ui.button(
        label="Random Example",
        on_change=lambda _: set_example_idx(random.randrange(len(imgs_dataset)) + 1),
    )
    return (random_example_btn,)


@app.cell
def __(get_example_idx, imgs_dataset, mo, set_example_idx):
    example_num = mo.ui.number(
        start=1,
        stop=len(imgs_dataset),
        step=1,
        value=get_example_idx(),
        label="Example:",
        on_change=set_example_idx,
    )
    return (example_num,)


@app.cell
def __(cls_dropdowns, mo, n_features):
    getter, setter = mo.state([0.0] * n_features * len(cls_dropdowns.value))
    return getter, setter


@app.cell
def __(beartype, getter, setter):
    @beartype.beartype
    def indexed_setter(i: int, v: float):
        setter(getter()[:i] + [v] + getter()[i + 1 :])

    return (indexed_setter,)


@app.cell
def __(features, functools, getter, indexed_setter, mo):
    sliders = [
        mo.ui.slider(
            start=-10,
            stop=10,
            step=0.1,
            value=getter()[i],
            on_change=functools.partial(indexed_setter, i),
        )
        for i, _ in enumerate(features)
    ]
    sliders = mo.ui.array(sliders)
    return (sliders,)


@app.cell
def __(example_num, mo, random_example_btn):
    mo.hstack([
        mo.hstack(
            [
                random_example_btn,
                example_num,
                # mo.md(f"'{classnames[target]}'"),
            ],
            justify="start",
        ),
        # cls_dropdown,
    ])
    return


@app.cell
def __(classnames, mo):
    def make_cls_dropdowns(n: int = 3):
        i_name_pairs = sorted(classnames.items(), key=lambda pair: pair[1])
        cls_dropdowns = []
        for i in range(n):
            dropdown = mo.ui.dropdown(
                options={name: j for j, name in i_name_pairs},
                label=f"Class {i + 1}:",
                value=i_name_pairs[i][1],
            )
            cls_dropdowns.append(dropdown)
        return mo.ui.array(cls_dropdowns)

    cls_dropdowns = make_cls_dropdowns()
    return cls_dropdowns, make_cls_dropdowns


@app.cell
def __(cls_dropdowns, mo):
    mo.vstack(cls_dropdowns)
    return


@app.cell
def __(sae, sparsity, torch):
    mask = torch.ones((sae.cfg.d_sae), dtype=bool)
    mask = mask & (sparsity < 1e-2)
    return (mask,)


@app.cell
def __(acts_dataset, einops, example_num, torch):
    acts_PD = torch.stack([
        acts_dataset[
            (example_num.value - 1) * acts_dataset.metadata.n_patches_per_img + i
        ]["act"]
        for i in range(acts_dataset.metadata.n_patches_per_img)
    ])
    acts_WHD = einops.rearrange(acts_PD, "(w h) d -> w h d", w=16, h=16)
    return acts_PD, acts_WHD


@app.cell
def __(acts_WHD, head, modify, torch):
    with torch.inference_mode():
        logits_WHC = head(acts_WHD)

        modified_acts_WHD = modify(acts_WHD)
        modified_logits_WHC = head(modified_acts_WHD)
    return logits_WHC, modified_acts_WHD, modified_logits_WHC


@app.cell
def __(cls_idxs, get_aggregate_features, mask):
    features = []
    for idxs in cls_idxs:
        features += get_aggregate_features(idxs, mask=mask)
    # features += get_random_features()
    return features, idxs


@app.cell
def __(
    classnames,
    cls_dropdowns,
    features,
    in1k_dataset,
    mo,
    n_features,
    n_images_per_feature,
    sliders,
    top_img_i,
):
    def make_sliders_ui():
        rows = []
        for slider, feature in zip(sliders, features):
            imgs, seen = [], set()
            for img_i in top_img_i[feature.latent].tolist():
                if img_i in seen:
                    continue
                imgs.append(in1k_dataset[img_i]["image"])
                seen.add(img_i)

                if len(seen) >= n_images_per_feature:
                    break

            row = [
                mo.hstack(
                    [slider, f"{slider.value:.3f}", f"Latent 12K/{feature.latent}"],
                    justify="start",
                ),
                mo.hstack(imgs, justify="start", gap=0.1),
            ]

            rows.append(mo.vstack(row))

        err_msg = f"len(rows) == {len(rows)} != n_features * len(cls_dropdowns.value) == {n_features * len(cls_dropdowns.value)}"
        assert len(rows) == n_features * len(cls_dropdowns.value), err_msg

        return mo.hstack(
            [
                mo.vstack(
                    [mo.md(f"Features for '{classnames[value]}'")]
                    + rows[n_features * i : n_features * (i + 1)]
                )
                for i, value in enumerate(cls_dropdowns.value)
            ],
            justify="start",
            gap=1.0,
        )

    make_sliders_ui()
    return (make_sliders_ui,)


@app.cell
def __(
    Image,
    example_num,
    imgs_dataset,
    logits_WHC,
    make_interpolated_pred,
    make_upsampled_pred,
    modified_logits_WHC,
    seg_to_img,
):
    display = {
        "Original Image": Image.fromarray(
            imgs_dataset[example_num.value - 1]["image"].numpy()
        ),
        "True Labels": seg_to_img(imgs_dataset[example_num.value - 1]["segmentation"]),
        "Predicted Labels (Upsampled)": seg_to_img(make_upsampled_pred(logits_WHC)),
        "Predicted Labels (Interpolated)": seg_to_img(
            make_interpolated_pred(logits_WHC)
        ),
        "Predicted Labels After Manipulation (Upsampled)": seg_to_img(
            make_upsampled_pred(modified_logits_WHC)
        ),
        "Predicted Labels After Manipulation (Interpolated)": seg_to_img(
            make_interpolated_pred(modified_logits_WHC)
        ),
    }
    return (display,)


@app.cell
def __(display, mo):
    mo.hstack(
        [
            mo.vstack([img, caption], align="center")
            for caption, img in list(display.items())
        ],
        widths="equal",
    )
    return


@app.cell
def __(beartype, dataclasses):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Feature:
        latent: int
        max_obs: float
        raw_obs: float

        @property
        def default(self) -> float:
            """Return raw_obs, scaled from [-10 * max_obs, 10 * max_obs] to [-1, 1]."""
            return self.scaled(self.raw_obs)

        def unscaled(self, x: float) -> float:
            """Scale from [-10, 10] to [10 * -max_obs, 10 * max_obs]."""

            return self.map_range(
                x,
                (-10, 10),
                (-10 * self.max_obs, 10 * self.max_obs),
            )

        def scaled(self, x: float) -> float:
            """Return x, scaled from [-10 * max_obs, 10 * max_obs] to [-10, 10]."""
            return self.map_range(
                x,
                (-10 * self.max_obs, 10 * self.max_obs),
                (-10, 10),
            )

        @staticmethod
        def map_range(
            x: float,
            domain: tuple[float | int, float | int],
            range: tuple[float | int, float | int],
        ):
            a, b = domain
            c, d = range
            if not (a <= x <= b):
                raise ValueError(f"x={x:.3f} must be in {[a, b]}.")
            return c + (x - a) * (d - c) / (b - a)

    return (Feature,)


@app.cell
def __(Int, Tensor, beartype, jaxtyped, torch):
    @jaxtyped(typechecker=beartype.beartype)
    def make_patch_lookup(
        *,
        patch_size_px: tuple[int, int],
        im_size_px: tuple[int, int] = (224, 224),
    ) -> Int[Tensor, "w_px h_px"]:
        im_w_px, im_h_px = im_size_px
        p_w_px, p_h_px = patch_size_px
        xv, yv = torch.meshgrid(torch.arange(im_w_px), torch.arange(im_h_px))
        patch_lookup = (xv // p_w_px) + (yv // p_h_px) * (im_h_px // p_h_px)
        return patch_lookup

    patch_lookup = make_patch_lookup(patch_size_px=(14, 14), im_size_px=(224, 224))
    return make_patch_lookup, patch_lookup


@app.cell
def __(
    acts_dataset,
    classnames,
    cls_dropdowns,
    imgs_dataset,
    mo,
    patch_lookup,
    torch,
):
    imgs_dataloader = torch.utils.data.DataLoader(
        imgs_dataset,
        num_workers=16,
        shuffle=False,
        batch_size=512,
        persistent_workers=True,
    )

    def get_cls_act_idx(cls: int):
        idxs = []
        for batch in mo.status.progress_bar(
            imgs_dataloader,
            remove_on_exit=True,
            title="Loading",
            subtitle=f"Getting class '{classnames[cls]}' segmentations.",
        ):
            batch_i, w_i, h_i, _ = (batch["segmentation"] == cls).nonzero(as_tuple=True)

            patch_i, image_i = torch.stack([
                patch_lookup[w_i, h_i],
                batch["index"][batch_i],
            ]).unique(dim=1)
            acts_i = image_i * acts_dataset.metadata.n_patches_per_img + patch_i

            idxs.append(acts_i)

        return torch.cat(idxs)

    cls_idxs = [get_cls_act_idx(v) for v in cls_dropdowns.value]
    return cls_idxs, get_cls_act_idx, imgs_dataloader


@app.cell
def __(
    Float,
    Tensor,
    beartype,
    einops,
    features,
    jaxtyped,
    sae,
    sliders,
    torch,
):
    @jaxtyped(typechecker=beartype.beartype)
    @torch.inference_mode
    def modify(
        acts_WHD: Float[Tensor, "width height d_vit"],
    ) -> Float[Tensor, "width height d_vit"]:
        x_hat_WHD, f_x_WHS, _ = sae(acts_WHD)

        err_WHD = acts_WHD - x_hat_WHD

        latents = [f.latent for f in features]
        values = torch.tensor([
            f.unscaled(sliders.value[i]) for i, f in enumerate(features)
        ])
        modified_f_x_WHS = f_x_WHS.clone()
        modified_f_x_WHS[..., latents] = values

        # Reproduce the SAE forward pass after f_x
        modified_x_hat_WHD = (
            einops.einsum(
                modified_f_x_WHS,
                sae.W_dec,
                "width height d_sae, d_sae d_vit -> width height d_vit",
            )
            + sae.b_dec
        )
        modified_WHD = err_WHD + modified_x_hat_WHD

        return modified_WHD

    return (modify,)


@app.cell
def __(
    Bool,
    Feature,
    Int,
    Tensor,
    acts_dataset,
    beartype,
    jaxtyped,
    mo,
    n_features,
    sae,
    top_values,
    torch,
):
    @jaxtyped(typechecker=beartype.beartype)
    @torch.inference_mode
    def get_aggregate_features(
        cls_act_idx: Int[Tensor, " n"],
        mask: Bool[Tensor, " d_sae"] = torch.ones(sae.cfg.d_sae, dtype=bool),
    ) -> list[Feature]:
        acts_MD = torch.stack([
            acts_dataset[i.item()]["act"]
            for i in mo.status.progress_bar(
                cls_act_idx,
                remove_on_exit=True,
                title="Loading",
                subtitle="Getting class-specific ViT acts",
            )
        ])

        max_examples = 1024 * 8

        # Shuffle
        acts_MD = acts_MD[torch.randperm(len(acts_MD))]
        # Select
        acts_MD = acts_MD[:max_examples]

        _, f_x_MS, _ = sae(acts_MD)

        f_x_S = f_x_MS.sum(axis=0)
        latents = torch.argsort(f_x_S, descending=True).cpu()

        latents = latents[mask[latents]][:n_features]

        max_obs = top_values[latents, 0]

        return [
            Feature(latent.item(), max.max().item(), 0.0)
            for latent, max in zip(latents, max_obs)
        ]

    return (get_aggregate_features,)


@app.cell
def __(csv, imgs_cfg, os):
    classnames = {}
    with open(os.path.join(imgs_cfg.root, "objectInfo150.txt")) as fd:
        for row in csv.DictReader(fd, delimiter="\t"):
            names = [name.strip() for name in row["Name"].split(",")]
            classnames[int(row["Idx"])] = ", ".join(names[:2])
    return classnames, fd, names, row


@app.cell
def __(Float, Tensor, Uint8, beartype, jaxtyped, torch):
    @jaxtyped(typechecker=beartype.beartype)
    @torch.inference_mode
    def make_upsampled_pred(
        logits_WHC: Float[Tensor, "width height classes"],
    ) -> Uint8[Tensor, "width height"]:
        return (
            torch.nn.functional.interpolate(
                logits_WHC.max(axis=-1).indices.view((1, 1, 16, 16)).float(),
                scale_factor=14,
            )
            .view((224, 224))
            .type(torch.uint8)
        )

    return (make_upsampled_pred,)


@app.cell
def __(Float, Tensor, Uint8, beartype, einops, jaxtyped, torch):
    @jaxtyped(typechecker=beartype.beartype)
    @torch.inference_mode
    def make_interpolated_pred(
        logits_WHC: Float[Tensor, "width height classes"],
    ) -> Uint8[Tensor, "width height"]:
        logits_CWH = einops.rearrange(
            logits_WHC, "width height classes -> classes width height"
        )
        upsampled_CWH = torch.nn.functional.interpolate(
            logits_CWH.contiguous().unsqueeze(0), size=(224, 224), mode="bilinear"
        )[0]
        pred_WH = upsampled_CWH.argmax(axis=0).cpu().type(torch.uint8)
        return pred_WH

    return (make_interpolated_pred,)


@app.cell
def __(Image, Tensor, UInt8, beartype, einops, jaxtyped, np, random):
    @jaxtyped(typechecker=beartype.beartype)
    def make_colors(seed: int = 42) -> UInt8[np.ndarray, "n 3"]:
        values = (0, 51, 102, 153, 204, 255)
        colors = []
        for r in values:
            for g in values:
                for b in values:
                    colors.append((r, g, b))
        random.Random(seed).shuffle(colors)
        colors = np.array(colors, dtype=np.uint8)

        return colors

    @jaxtyped(typechecker=beartype.beartype)
    def seg_to_img(map: UInt8[Tensor, "width height *channel"]) -> Image.Image:
        map = map.numpy()
        if map.ndim == 3:
            map = einops.rearrange(map, "w h () -> w h")
        colored = np.zeros((224, 224, 3), dtype=np.uint8)
        for i, color in enumerate(make_colors()):
            colored[map == i + 1, :] = color

        return Image.fromarray(colored)

    return make_colors, seg_to_img


@app.cell
def __(saev, v2):
    in1k_dataset = saev.activations.get_dataset(
        saev.config.ImagenetDataset(),
        img_transform=v2.Compose([
            v2.Resize(size=(128, 128)),
            v2.CenterCrop(size=(112, 112)),
        ]),
    )
    return (in1k_dataset,)


@app.cell
def __():
    import sys

    pkg_root = "/home/stevens.994/projects/saev"
    if pkg_root not in sys.path:
        sys.path.append(pkg_root)

    import csv
    import dataclasses
    import functools
    import os.path
    import random

    import beartype
    import einops
    import einops.layers.torch
    import marimo as mo
    import numpy as np
    import torch
    from jaxtyping import Bool, Float, Int, UInt8, jaxtyped
    from PIL import Image
    from torchvision.transforms import v2

    import contrib.semseg.training
    import saev.activations
    import saev.config

    return (
        Bool,
        Float,
        Image,
        Int,
        UInt8,
        beartype,
        contrib,
        csv,
        dataclasses,
        einops,
        functools,
        jaxtyped,
        mo,
        np,
        os,
        pkg_root,
        random,
        saev,
        sys,
        torch,
        v2,
    )


if __name__ == "__main__":
    app.run()
