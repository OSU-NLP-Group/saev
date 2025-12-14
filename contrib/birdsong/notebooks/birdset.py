import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import datasets
    import marimo as mo
    import logging

    import pathlib
    import soundfile as sf
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import saev.data.datasets
    import torchaudio
    import torch
    import torch.nn.functional as F
    from torchaudio.compliance.kaldi import fbank

    from jaxtyping import Float, jaxtyped
    import beartype
    from torch import Tensor

    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    return Float, Tensor, np, pathlib, plt, saev, torch


@app.cell
def _():
    import matplotlib as mpl

    colormap = mpl.colormaps.get_cmap("plasma")
    return


@app.cell
def _(pathlib):
    root = pathlib.Path("/fs/ess/PAS2136/samuelstevens/datasets/birdclef-2025")
    return (root,)


@app.cell
def _(root, saev):
    cfg = saev.data.datasets.BirdClef2025(root)
    birdclef_ds = saev.data.datasets.BirdClef2025Dataset(
        cfg, audio_transform=saev.data.bird_mae.transform
    )
    return (birdclef_ds,)


@app.cell
def _(pathlib, saev):
    birdclef = saev.data.IndexedDataset(
        saev.data.IndexedConfig(
            pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards/5e37a03c"),
            layer=13,
            debug=True,
        )
    )
    birdclef_dl = saev.data.ShuffledDataLoader(
        saev.data.ShuffledConfig(
            pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards/5e37a03c"),
            layer=13,
            batch_size=1024 * 4,
            debug=True,
            min_buffer_fill=0.2,
        )
    )
    return (birdclef_dl,)


@app.cell
def _(pathlib, saev):
    ade20k = saev.data.IndexedDataset(
        saev.data.IndexedConfig(
            pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"),
            layer=13,
            debug=True,
        )
    )
    ade20k_dl = saev.data.ShuffledDataLoader(
        saev.data.ShuffledConfig(
            pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards/614861a0"),
            layer=13,
            batch_size=1024 * 4,
            debug=True,
            min_buffer_fill=0.2,
        )
    )
    return (ade20k_dl,)


@app.cell
def _(ade20k_dl, birdclef_dl, saev, torch):
    def _(dl, *, total: int = 300_000):
        n_seen = 0
        acts = []

        for batch in saev.helpers.progress(
            dl, every=10, total=total // dl.batch_size + 1
        ):
            bsz, _ = batch["act"].shape
            acts.append(batch["act"])

            n_seen += bsz
            if n_seen >= total:
                break

        return torch.concat(acts)[:total]

    birdclef_acts = _(birdclef_dl)
    ade20k_acts = _(ade20k_dl)

    birdclef_acts.shape, ade20k_acts.shape
    return ade20k_acts, birdclef_acts


@app.cell
def _(ade20k_acts, birdclef_acts, np, plt):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        xlow, xhigh = -1000, 5000
        ax.hist(
            birdclef_acts.flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE (BirdClef)",
        )
        ax.hist(
            ade20k_acts.flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:orange",
            alpha=0.5,
            label="DINOv3 (ADE20K)",
        )
        ax.set_yscale("log")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)

        return fig

    _()
    return


@app.cell
def _(ade20k_acts, birdclef_acts, np, plt):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        xlow, xhigh = -750, 750
        ax.hist(
            birdclef_acts.flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE (BirdClef)",
        )
        ax.hist(
            ade20k_acts.flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:orange",
            alpha=0.5,
            label="DINOv3 (ADE20K)",
        )
        ax.set_yscale("log")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)

        return fig

    _()
    return


@app.cell
def _(ade20k_acts, birdclef_acts, np, plt):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        xlow, xhigh = -100, 100
        ax.hist(
            birdclef_acts.flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE (BirdClef)",
        )
        ax.hist(
            ade20k_acts.flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:orange",
            alpha=0.5,
            label="DINOv3 (ADE20K)",
        )
        ax.set_yscale("log")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)

        return fig

    _()
    return


@app.cell
def _(ade20k_acts, birdclef_acts, np, plt, torch):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        xlow, xhigh = 0, 5000
        ax.hist(
            torch.linalg.norm(birdclef_acts, ord=2, dim=1),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE (BirdClef)",
        )
        ax.hist(
            torch.linalg.norm(ade20k_acts, ord=2, dim=1),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:orange",
            alpha=0.5,
            label="DINOv3 (ADE20K)",
        )
        ax.set_yscale("log")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)
        ax.set_xlabel("L2 Norm of Activation Vectors")

        return fig

    _()
    return


@app.cell
def _(birdclef_acts, np, plt, torch):
    # What dimension consistently has a huge value?
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        sample_i, d_i = torch.where(birdclef_acts > 1000)
        xlow, xhigh = 0, 1024
        ax.hist(
            d_i,
            bins=np.linspace(xlow, xhigh, 1025),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE (BirdClef)",
        )
        # ax.hist(torch.linalg.norm(ade20k_acts, ord=2, dim=1), bins=np.linspace(xlow, xhigh, 100), color='tab:orange', alpha=0.5, label='DINOv3 (ADE20K)')
        print(torch.all(d_i == 295))
        ax.set_yscale("log")
        # ax.set_ylim(0, 500_000)
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)
        ax.set_xlabel("Dimensions")

        return fig

    _()
    return


@app.cell
def _(birdclef_acts, np, plt):
    # What dimension consistently has a huge value?
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        xlow, xhigh = 2500, 4500
        ax.hist(
            birdclef_acts[:, 295],
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE channel=296/1024 (BirdClef)",
        )
        # ax.hist(torch.linalg.norm(ade20k_acts, ord=2, dim=1), bins=np.linspace(xlow, xhigh, 100), color='tab:orange', alpha=0.5, label='DINOv3 (ADE20K)')
        ax.set_yscale("log")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)
        # ax.set_xlabel('L2 Norm of Activation Vectors')

        return fig

    _()
    return


@app.cell
def _(birdclef_acts):
    birdclef_acts.shape
    return


@app.cell
def _(ade20k_acts, birdclef_acts, np, plt, torch):
    def _():
        fig, ax = plt.subplots(dpi=200, layout="constrained")
        xlow, xhigh = -1000, 5000
        ax.hist(
            torch.cat((
                birdclef_acts[:, :295].flatten(),
                birdclef_acts[:, 296:].flatten(),
            )),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:blue",
            alpha=0.5,
            label="BirdMAE (BirdClef)",
        )
        ax.hist(
            torch.cat((ade20k_acts[:, :295].flatten(), ade20k_acts[:, 296:].flatten())),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:orange",
            alpha=0.5,
            label="DINOv3 (ADE20K)",
        )
        ax.hist(
            birdclef_acts[:, 295].flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:red",
            alpha=0.5,
            label="BirdMAE (BirdClef) ($d_\\text{bad}$)",
        )
        ax.hist(
            ade20k_acts[:, 295].flatten(),
            bins=np.linspace(xlow, xhigh, 100),
            color="tab:green",
            alpha=0.5,
            label="DINOv3 (ADE20K) ($d_\\text{bad}$)",
        )
        ax.set_yscale("log")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(xlow, xhigh)

        return fig

    _()
    return


@app.cell
def _(birdclef_acts):
    birdclef_acts.mean()
    return


@app.cell
def _(saev, torch):
    model = saev.data.bird_mae.Transformer("Bird-MAE-Large").eval().to(torch.float32)
    return (model,)


@app.cell
def _(model, np, plt):
    def _():
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, dpi=200, sharey=True, layout="constrained"
        )
        bounds_b = (-1.5, 1.5)
        bounds_w = (-0.3, 3)
        for i, block in enumerate(model.model.blocks):
            ax1.hist(
                block.norm1.weight.data.numpy(),
                np.linspace(*bounds_w, 100),
                alpha=0.05,
                color="tab:red",
                label=f"b{i}-norm1",
            )
            ax2.hist(
                block.norm1.bias.data.numpy(),
                np.linspace(*bounds_b, 100),
                alpha=0.05,
                color="tab:red",
                label=f"b{i}-norm1",
            )
            ax1.hist(
                block.norm2.weight.data.numpy(),
                np.linspace(*bounds_w, 100),
                alpha=0.05,
                color="tab:blue",
                label=f"b{i}-norm2",
            )
            ax2.hist(
                block.norm2.bias.data.numpy(),
                np.linspace(*bounds_b, 100),
                alpha=0.05,
                color="tab:blue",
                label=f"b{i}-norm2",
            )
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        # ax1.legend()
        # ax2.legend()

        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        # ax1.set_xlim(xlow, xhigh)
        # ax2.set_xlim(xlow, xhigh)

        ax1.set_title("LayerNorm Weight")
        ax2.set_title("LayerNorm Bias")
        # ax.set_xlabel('L2 Norm of Activation Vectors')

        return fig

    _()
    return


@app.cell
def _(model):
    def _():
        for i, block in enumerate(model.model.blocks):
            print(i, block.norm1.weight.data.max().item())
            print(i, block.norm1.bias.data.max().item())
            print(i, block.norm2.weight.data.max().item())
            print(i, block.norm2.bias.data.max().item())

    _()
    return


@app.cell
def _():
    d_bad = 295
    return (d_bad,)


@app.cell
def _(Float, Tensor, birdclef_ds, model, torch):
    def make_graph1_hook(i: int, storage):
        def hook(
            module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
        ) -> None:
            storage[i] = args[0][0]
            storage[i + 1] = output[0]
            print(i)

        return hook

    def get_graph1_inputs(item: int = 0):
        storage = torch.zeros((len(model.model.blocks) + 1, 257, 1024))
        # handles = [model.model.patch_embed.register_forward_hook(make_hook(i, storage))]
        handles = []
        for i, block in enumerate(model.model.blocks):
            handles.append(block.register_forward_hook(make_graph1_hook(i, storage)))
            # handles.append(block.drop_path2.register_forward_hook(make_hook(i * 2 + 1, storage)))

        model(birdclef_ds[item]["data"][None, ...])

        for handle in handles:
            handle.remove()

        return storage

    block_acts = get_graph1_inputs(item=3)
    return (block_acts,)


@app.cell
def _(block_acts, d_bad, np, plt, torch):
    def graph1():
        fig, ax1 = plt.subplots(dpi=200, sharey=True, layout="constrained")
        x = np.arange(25)

        data = block_acts.detach()
        handles = []

        mean = data.mean(dim=(1, 2))

        (handle,) = ax1.plot(
            x, mean, marker="o", color="tab:blue", alpha=0.5, label="Mean (L)"
        )
        handles.append(handle)

        mean_without_d_bad = torch.cat(
            (data[:, :, :d_bad], data[:, :, d_bad + 1 :]), dim=2
        ).mean(dim=(1, 2))
        (handle,) = ax1.plot(
            x,
            mean_without_d_bad,
            marker="s",
            color="tab:blue",
            alpha=0.5,
            label="Mean w/o $d_\\text{bad}$ (R)",
        )
        handles.append(handle)

        # std = torch.cat((data[:, :, :d_bad], data[:, :, d_bad+1:]), dim=2).std(dim=(1,2))
        # ax1.fill_between(x, mean_without_d_bad + std, mean_without_d_bad - std, color='tab:blue', alpha=0.5)

        ax2 = ax1.twinx()
        mean_d_bad = data[:, :, d_bad].mean(dim=(1))
        (handle,) = ax2.plot(
            x,
            mean_d_bad,
            marker="*",
            color="tab:green",
            alpha=0.5,
            label="Mean $d_\\text{bad}$ (L)",
        )
        handles.append(handle)

        std = data[:, :, d_bad].std(dim=(1))
        ax2.fill_between(
            x, mean_d_bad + std, mean_d_bad - std, color="tab:green", alpha=0.5
        )

        # norm_without_d_bad = torch.linalg.norm(torch.cat((data[:, :, :d_bad], data[:, :, d_bad+1:]), dim=2), ord=2, dim=2).mean(dim=1)
        # (handle,) = ax2.plot(x, norm_without_d_bad, marker='^', color='tab:red', label='Norm w/o $d_\\text{bad}$')
        # handles.append(handle)

        # ax.fill_between(x, mean + std, mean - std, alpha=0.2)
        ax1.set_xticks(x, [str(i) if i % 4 == 0 else "" for i in x.tolist()])
        ax1.legend(handles=handles, loc="center right")

        # ax1.tick_params(axis="y", colors="tab:blue")
        ax1.set_ylabel("Mean Activation Value", color="tab:blue")

        ax2.set_ylabel("Mean Activation Value ($d_\\text{bad}$)", color="tab:green")
        # ax2.set_yscale("symlog")

        return fig

    graph1()
    return


@app.cell
def _(block_acts, d_bad, np, plt):
    def graph1_extra():
        fig, ax1 = plt.subplots(dpi=200, sharey=True, layout="constrained")
        x = np.arange(25)

        data = block_acts.detach()[:, 1:, :]

        mean = data.mean(dim=1)

        for d in range(1024):
            if d == d_bad:
                continue

            else:
                ax1.plot(x, mean[:, d], color="tab:blue", alpha=0.03, clip_on=False)

        ax1.plot([], [], color="tab:blue", label="Mean")

        ax1.plot(
            x,
            mean[:, d_bad],
            marker="o",
            color="tab:red",
            label="Mean $d_\\text{bad}$",
            clip_on=False,
        )
        print(mean[:, d_bad])

        # ax.fill_between(x, mean + std, mean - std, alpha=0.2)
        ax1.set_xticks(x, [str(i) if i % 4 == 0 else "" for i in x.tolist()])
        ax1.legend()

        # ax1.tick_params(axis="y", colors="tab:blue")
        ax1.set_ylabel("Neuron Activation Value")
        ax1.set_xlabel("Block")

        ax1.spines[["top", "right"]].set_visible(False)
        ax1.set_yscale("symlog")

        ax1.set_title(
            "Block Outputs (Graph #1)", color=tuple(c / 256 for c in (65, 144, 107))
        )

        return fig

    graph1_extra()
    return


@app.cell
def _(Float, Tensor, birdclef_ds, model, torch):
    def make_graph2_hook(i: int, storage):
        def hook(
            module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
        ) -> None:
            storage[i] = output[0]
            print(i)

        return hook

    def get_graph2_inputs(item: int = 0):
        storage = torch.zeros((len(model.model.blocks), 257, 1024))
        # handles = [model.model.patch_embed.register_forward_hook(make_hook(i, storage))]
        handles = []
        for i, block in enumerate(model.model.blocks):
            handles.append(
                block.attn.register_forward_hook(make_graph2_hook(i, storage))
            )
            # handles.append(block.drop_path2.register_forward_hook(make_hook(i * 2 + 1, storage)))

        model(birdclef_ds[item]["data"][None, ...])

        for handle in handles:
            handle.remove()

        return storage

    attn_acts = get_graph2_inputs(item=3)
    return (attn_acts,)


@app.cell
def _(attn_acts, d_bad, np, plt, torch):
    def graph2():
        fig, ax1 = plt.subplots(dpi=200, sharey=True, layout="constrained")
        x = np.arange(24)

        data = attn_acts.detach()
        handles = []

        mean = data.mean(dim=(1, 2))

        (handle,) = ax1.plot(
            x, mean, marker="o", color="tab:blue", alpha=0.5, label="Mean (L)"
        )
        handles.append(handle)

        mean_without_d_bad = torch.cat(
            (data[:, :, :d_bad], data[:, :, d_bad + 1 :]), dim=2
        ).mean(dim=(1, 2))
        (handle,) = ax1.plot(
            x,
            mean_without_d_bad,
            marker="s",
            color="tab:blue",
            alpha=0.5,
            label="Mean w/o $d_\\text{bad}$ (R)",
        )
        handles.append(handle)

        std = torch.cat((data[:, :, :d_bad], data[:, :, d_bad + 1 :]), dim=2).std(
            dim=(1, 2)
        )
        ax1.fill_between(
            x,
            mean_without_d_bad + std,
            mean_without_d_bad - std,
            color="tab:blue",
            alpha=0.5,
        )

        ax2 = ax1  # .twinx()
        mean_d_bad = data[:, :, d_bad].mean(dim=(1))
        (handle,) = ax2.plot(
            x,
            mean_d_bad,
            marker="*",
            color="tab:green",
            alpha=0.5,
            label="Mean $d_\\text{bad}$ (L)",
        )
        handles.append(handle)

        std = data[:, :, d_bad].std(dim=(1))
        ax2.fill_between(
            x, mean_d_bad + std, mean_d_bad - std, color="tab:green", alpha=0.5
        )

        # norm_without_d_bad = torch.linalg.norm(torch.cat((data[:, :, :d_bad], data[:, :, d_bad+1:]), dim=2), ord=2, dim=2).mean(dim=1)
        # (handle,) = ax2.plot(x, norm_without_d_bad, marker='^', color='tab:red', label='Norm w/o $d_\\text{bad}$')
        # handles.append(handle)

        # ax.fill_between(x, mean + std, mean - std, alpha=0.2)
        ax1.set_xticks(x, [str(i) if i % 4 == 0 else "" for i in x.tolist()])
        ax1.legend(handles=handles, loc="center right")

        # ax1.tick_params(axis="y", colors="tab:blue")
        ax1.set_ylabel("Mean Activation Value", color="tab:blue")

        ax2.set_ylabel("Mean Activation Value ($d_\\text{bad}$)", color="tab:green")
        # ax2.set_yscale("symlog")
        ax1.set_xlabel("Attn. Layer")

        return fig

    graph2()
    return


@app.cell
def _(attn_acts, d_bad, np, plt):
    def graph2_extra():
        fig, ax1 = plt.subplots(dpi=200, sharey=True, layout="constrained")
        x = np.arange(24)

        data = attn_acts.detach()[:, 1:, :]

        mean = data.mean(dim=1)

        for d in range(1024):
            if d == d_bad:
                continue

            else:
                ax1.plot(x, mean[:, d], color="tab:blue", alpha=0.03, clip_on=False)

        ax1.plot([], [], color="tab:blue", label="Mean")

        ax1.plot(
            x,
            mean[:, d_bad],
            marker="o",
            color="tab:red",
            label="Mean $d_\\text{bad}$",
            clip_on=False,
        )
        print(mean[:, d_bad])

        # ax.fill_between(x, mean + std, mean - std, alpha=0.2)
        ax1.set_xticks(x, [str(i + 1) if i % 4 == 0 else "" for i in x.tolist()])
        ax1.legend()

        # ax1.tick_params(axis="y", colors="tab:blue")
        ax1.set_ylabel("Neuron Activation Value")
        ax1.set_xlabel("Attention Layer")

        ax1.spines[["top", "right"]].set_visible(False)
        ax1.set_yscale("symlog")
        ax1.set_title(
            "Attention Outputs (Graph #2)", color=tuple(c / 256 for c in (231, 175, 93))
        )

        return fig

    graph2_extra()
    return


@app.cell
def _(Float, Tensor, birdclef_ds, model, torch):
    def make_graph3_hook(i: int, storage):
        def hook(
            module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
        ) -> None:
            storage[i] = output[0]
            print(i)

        return hook

    def get_graph3_inputs(item: int = 0):
        storage = torch.zeros((len(model.model.blocks), 257, 1024))
        # handles = [model.model.patch_embed.register_forward_hook(make_hook(i, storage))]
        handles = []
        for i, block in enumerate(model.model.blocks):
            handles.append(
                block.mlp.register_forward_hook(make_graph3_hook(i, storage))
            )
            # handles.append(block.drop_path2.register_forward_hook(make_hook(i * 2 + 1, storage)))

        model(birdclef_ds[item]["data"][None, ...])

        for handle in handles:
            handle.remove()

        return storage

    mlp_acts = get_graph3_inputs(item=3)
    return (mlp_acts,)


@app.cell
def _(d_bad, mlp_acts, np, plt):
    def graph3_extra():
        fig, ax1 = plt.subplots(dpi=200, sharey=True, layout="constrained")
        x = np.arange(24)

        data = mlp_acts.detach()

        mean = data.mean(dim=1)

        for d in range(1024):
            if d == d_bad:
                continue

            else:
                ax1.plot(x, mean[:, d], color="tab:blue", alpha=0.03, clip_on=False)

        ax1.plot([], [], color="tab:blue", label="Mean")

        ax1.plot(
            x,
            mean[:, d_bad],
            marker="o",
            color="tab:red",
            label="Mean $d_\\text{bad}$",
            clip_on=False,
        )
        print(mean[:, d_bad])

        # ax.fill_between(x, mean + std, mean - std, alpha=0.2)
        ax1.set_xticks(x, [str(i + 1) if i % 4 == 0 else "" for i in x.tolist()])
        ax1.legend()

        # ax1.tick_params(axis="y", colors="tab:blue")
        ax1.set_ylabel("Neuron Activation Value")
        ax1.set_xlabel("MLP Layer")

        ax1.spines[["top", "right"]].set_visible(False)
        ax1.set_yscale("symlog")
        ax1.set_title(
            "MLP Outputs (Graph #3)", color=tuple(c / 256 for c in (75, 100, 225))
        )

        return fig

    graph3_extra()
    return


@app.cell
def _(Float, Tensor, birdclef_ds, model, torch):
    def make_graph4_hook(i: int, storage):
        def hook(
            module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
        ) -> None:
            storage[0, i] = args[0][0]
            storage[1, i] = output[0]
            print(i)

        return hook

    def get_graph4_inputs(item: int = 0):
        storage = torch.zeros((2, len(model.model.blocks), 257, 1024))
        handles = []
        for i, block in enumerate(model.model.blocks):
            handles.append(
                block.norm2.register_forward_hook(make_graph4_hook(i, storage))
            )

        model(birdclef_ds[item]["data"][None, ...])

        for handle in handles:
            handle.remove()

        return storage

    norm2_acts = get_graph4_inputs(item=3)
    return (norm2_acts,)


@app.cell
def _(d_bad, norm2_acts, np, plt):
    def graph4_extra():
        fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=200, layout="constrained")
        x = np.arange(24)

        data = norm2_acts[:, :, 1:].detach()

        mean = data.mean(dim=2)
        print(data.mean(dim=(2, 3)))
        print(data.shape, mean.shape)

        for d in range(1024):
            if d == d_bad:
                continue

            ax1.plot(x, mean[0, :, d], color="tab:blue", alpha=0.03, clip_on=False)
            ax2.plot(x, mean[1, :, d], color="tab:blue", alpha=0.03, clip_on=False)

        ax1.plot([], [], color="tab:blue", label="Others")
        ax2.plot([], [], color="tab:blue", label="Others")

        ax1.plot(
            x,
            mean[0, :, d_bad],
            marker="o",
            color="tab:red",
            label="$d_\\text{bad}$",
            clip_on=False,
        )
        ax2.plot(
            x,
            mean[1, :, d_bad],
            marker="o",
            color="tab:red",
            label="$d_\\text{bad}$",
            clip_on=False,
        )

        # ax.fill_between(x, mean + std, mean - std, alpha=0.2)
        ax2.set_xticks(x, [str(i + 1) if i % 4 == 0 else "" for i in x.tolist()])
        ax1.legend(loc="lower left")
        ax2.legend(loc="lower left")

        # ax1.tick_params(axis="y", colors="tab:blue")
        ax1.set_ylabel("Neuron Magnitude")
        ax2.set_ylabel("Neuron Magnitude")
        ax2.set_xlabel("Layer")

        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        ax1.set_yscale("symlog")
        ax2.set_yscale("symlog", linthresh=0.015)

        ax1.set_title("Inputs")
        ax2.set_title("Outputs")
        fig.suptitle("LayerNorm 2")

        return fig

    graph4_extra()
    return


@app.cell
def _(d_bad, model, np, plt):
    def graph_layernorm2():
        fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=200, layout="constrained")
        xs = np.arange(24)
        ys_all = np.zeros((2, 24, 1024))

        for i, block in enumerate(model.model.blocks):
            ys_all[0, i] = block.norm2.weight.data.detach().numpy()
            ys_all[1, i] = block.norm2.bias.data.detach().numpy()

        for d in range(1024):
            if d == d_bad:
                continue

            ax1.plot(xs, ys_all[0, :, d], color="tab:blue", alpha=0.03, clip_on=False)
            ax2.plot(xs, ys_all[1, :, d], color="tab:blue", alpha=0.03, clip_on=False)

        ax1.plot([], [], color="tab:blue", label="Others")
        ax2.plot([], [], color="tab:blue", label="Others")

        ax1.plot(
            xs,
            ys_all[0, :, d_bad],
            marker="o",
            color="tab:red",
            label="$d_\\text{bad}$",
            clip_on=False,
        )
        ax2.plot(
            xs,
            ys_all[1, :, d_bad],
            marker="o",
            color="tab:red",
            label="$d_\\text{bad}$",
            clip_on=False,
        )

        ax2.set_xticks(xs, [str(i + 1) if i % 4 == 0 else "" for i in xs.tolist()])
        ax1.legend(loc="lower left")
        ax2.legend(loc="lower left")

        # ax1.tick_params(axis="y", colors="tab:blue")
        # ax1.set_ylabel("Weight Magnitude")
        # ax2.set_ylabel("Bias Magnitude")
        ax2.set_xlabel("Layer")

        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        # ax1.set_yscale('symlog')
        # ax2.set_yscale('symlog', linthresh=0.015)

        ax1.set_title("Weight")
        ax2.set_title("Bias")
        fig.suptitle("LayerNorm 2 Params")

        return fig

    graph_layernorm2()
    return


@app.cell
def _(d_bad, model, np, plt):
    def graph_layernorm1():
        fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=200, layout="constrained")
        xs = np.arange(24)
        ys_all = np.zeros((2, 24, 1024))

        for i, block in enumerate(model.model.blocks):
            ys_all[0, i] = block.norm1.weight.data.detach().numpy()
            ys_all[1, i] = block.norm1.bias.data.detach().numpy()

        for d in range(1024):
            if d == d_bad:
                continue

            ax1.plot(xs, ys_all[0, :, d], color="tab:blue", alpha=0.03, clip_on=False)
            ax2.plot(xs, ys_all[1, :, d], color="tab:blue", alpha=0.03, clip_on=False)

        ax1.plot([], [], color="tab:blue", label="Others")
        ax2.plot([], [], color="tab:blue", label="Others")

        ax1.plot(
            xs,
            ys_all[0, :, d_bad],
            marker="o",
            color="tab:red",
            label="$d_\\text{bad}$",
            clip_on=False,
        )
        ax2.plot(
            xs,
            ys_all[1, :, d_bad],
            marker="o",
            color="tab:red",
            label="$d_\\text{bad}$",
            clip_on=False,
        )

        ax2.set_xticks(xs, [str(i + 1) if i % 4 == 0 else "" for i in xs.tolist()])
        ax1.legend(loc="lower left")
        ax2.legend(loc="lower left")

        # ax1.tick_params(axis="y", colors="tab:blue")
        # ax1.set_ylabel("Weight Magnitude")
        # ax2.set_ylabel("Bias Magnitude")
        ax2.set_xlabel("Layer")

        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        # ax1.set_yscale('symlog')
        # ax2.set_yscale('symlog', linthresh=0.015)

        ax1.set_title("Weight")
        ax2.set_title("Bias")
        fig.suptitle("LayerNorm 1 Params")

        return fig

    graph_layernorm1()
    return


@app.cell
def _(model):
    model.model
    return


@app.cell
def _(model, torch):
    torch.linalg.norm(model.model.blocks[0].mlp.fc2.weight.data, dim=1).max()
    return


@app.cell
def _(d_bad, model, np, plt, torch):
    def _():
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, dpi=200, sharey=False, layout="constrained"
        )

        xs = np.arange(24)
        ys_all = np.zeros((2, 24, 1024))

        for i, block in enumerate(model.model.blocks):
            norm = torch.linalg.norm(block.mlp.fc2.weight.data, dim=1).detach().numpy()
            ys_all[0, i] = norm
            ys_all[1, i] = block.mlp.fc2.bias.data.detach().numpy()

        for d in np.arange(1024):
            if d == d_bad:
                continue

            ax1.plot(xs, ys_all[0, :, d], alpha=0.1, color="tab:blue")
            ax2.plot(xs, ys_all[1, :, d], alpha=0.1, color="tab:blue")

        ax1.plot(
            xs,
            ys_all[0, :, d_bad],
            marker="o",
            color="tab:red",
            label="Column $d_\\text{bad}$ of fc2.w",
        )
        ax2.plot(
            xs,
            ys_all[1, :, d_bad],
            marker="o",
            color="tab:red",
            label="Column $d_\\text{bad}$ of fc2.b",
        )

        ax1.plot([], [], color="tab:blue", label="Other columns of fc2.w")
        ax2.plot([], [], color="tab:blue", label="Other columns of fc2.b")

        ax1.legend(loc="lower left")
        ax2.legend(loc="lower left")

        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        ax1.set_ylabel("L2 Norm")
        ax2.set_ylabel("Magnitude")

        ax2.set_xlabel("Layer")
        ax1.set_yscale("log")

        ax1.set_title("fc2.weight L2 Norm")
        ax2.set_title("fc2.bias Magnitude")

        fig.suptitle(
            "fc2 Parameters (Graph #1)", color=tuple(c / 256 for c in (65, 144, 107))
        )

        return fig

    _()
    return


@app.cell
def _(d_bad, model, np, plt, torch):
    def _():
        fig, ax1 = plt.subplots(nrows=1, dpi=200, sharey=True, layout="constrained")

        xs = np.arange(24)
        ys_all = np.zeros((24, 1024))

        for i, block in enumerate(model.model.blocks):
            norm = torch.linalg.norm(block.mlp.fc1.weight.data, dim=0).detach().numpy()
            # print(np.arange(d_bad).shape, norm[:d_bad].shape)
            ys_all[i] = norm
            # ys_all[1, i] = block.mlp.fc1.bias.data.detach().numpy()

            # ys_other[i, :d_bad] = norm[:d_bad]
            # ys_other[i, d_bad:] = norm[d_bad+1:]

        (handle,) = ax1.plot(
            xs,
            ys_all[:, d_bad],
            marker="o",
            color="tab:red",
            label="$||\\cdot||_2$ of fc1.w ($d_\\text{bad}$)",
        )
        for d in np.arange(1024):
            if d == d_bad:
                continue
            label = None
            if d == 0:
                label = "$||\\cdot||_2$ of fc1.w"
            ax1.plot(xs, ys_all[:, d], alpha=0.03, color="tab:blue", label=label)

        ax1.legend(handles=[handle])
        # ax2.legend()

        ax1.spines[["top", "right"]].set_visible(False)
        # ax2.spines[["top", "right"]].set_visible(False)

        # ax1.set_xlim(xlow, xhigh)
        # ax2.set_xlim(xlow, xhigh)

        # ax1.set_title("LayerNorm Weight")
        # ax2.set_title("LayerNorm Bias")
        ax1.set_ylabel("L2 Norm")
        ax1.set_xlabel("Layer")
        ax1.set_yscale("log")

        return fig

    _()
    return


@app.cell
def _(d_bad, model, np, plt):
    def _():
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, dpi=200, sharey=True, layout="constrained"
        )

        xs = np.arange(24)
        ys_all = np.zeros((2, 24, 1024))

        for i, block in enumerate(model.model.blocks):
            ys_all[0, i] = block.norm2.weight.data
            ys_all[1, i] = block.norm2.bias.data

        for d in np.arange(1024):
            if d == d_bad:
                continue

            ax1.plot(xs, ys_all[0, :, d], alpha=0.03, color="tab:blue")
            ax2.plot(xs, ys_all[1, :, d], alpha=0.03, color="tab:blue")

        ax1.plot(
            xs,
            ys_all[0, :, d_bad],
            marker="o",
            color="tab:red",
            label="Dim $d_\\text{bad}$ of norm2.w",
        )
        ax2.plot(
            xs,
            ys_all[1, :, d_bad],
            marker="o",
            color="tab:red",
            label="Dim $d_\\text{bad}$ of norm2.b",
        )

        ax1.plot([], [], color="tab:blue", label="Other dims of norm2.w")
        ax2.plot([], [], color="tab:blue", label="Other dims of norm2.b")

        ax1.legend()
        ax2.legend()

        ax1.spines[["top", "right"]].set_visible(False)
        ax2.spines[["top", "right"]].set_visible(False)

        ax1.set_ylabel("Weight Value")
        ax2.set_ylabel("Bias Value")

        ax2.set_xlabel("Layer")
        # ax1.set_yscale('symlog')
        ax1.set_title("")

        return fig

    _()
    return


@app.cell
def _(model):
    model.model.blocks[0]
    return


@app.cell
def _(model, plt):
    def _():
        fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=300, layout="constrained")
        ax1.imshow(model.model.blocks[0].mlp.fc1.weight.data.detach().log10().T.numpy())
        ax2.imshow(model.model.blocks[0].mlp.fc2.weight.data.detach().log10().numpy())
        return fig

    _()
    return


@app.cell
def _(model):
    W = model.model.blocks[0].mlp.fc2.weight  # expected shape (1024, 4096)
    b = model.model.blocks[0].mlp.fc2.bias  # (1024,)

    row_norms = W.norm(dim=1)  # per output dim
    col_norms = W.norm(dim=0)  # per hidden dim

    row_norms[295], col_norms[295], row_norms.median(), col_norms.median()
    return


@app.cell
def _(model):
    model.model.blocks[0].mlp.fc1.weight.data.shape
    return


if __name__ == "__main__":
    app.run()
