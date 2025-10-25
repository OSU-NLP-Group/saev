import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import polars as pl

    import saev.data
    import saev.data.datasets

    return pathlib, pl, saev


@app.cell
def _(pathlib):
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return (shards_root,)


@app.cell
def _(pathlib, pl, saev, shards_root):
    def format_ckpt(family: str, ckpt: str) -> str:
        if family == "dinov3":
            name_ds, sha = pathlib.Path(ckpt).stem.split("-")
            _, name, pretrain, ds = name_ds.split("_")
            assert pretrain == "pretrain"
            if name == "vitl16":
                return "ViT-L/16"
            if name == "vitb16":
                return "ViT-B/16"
            if name == "vits16":
                return "ViT-S/16"

            raise ValueError(name)

        if family == "clip":
            arch, pretrained = ckpt.split("/")
            if pretrained == "openai":
                return arch

            return ckpt

        raise ValueError(family)

    def format_data_cfg(data: saev.data.datasets.DatasetConfig) -> dict[str, object]:
        if isinstance(data, saev.data.datasets.Cifar10):
            return {"name": "CIFAR10"}

        if isinstance(data, saev.data.datasets.ImgSegFolder):
            if "ADE" in str(data.root):
                return {"name": "ADE20K", "split": data.split}

            if "fish-vista" in str(data.root):
                return {"name": "FistVista (Seg)", "split": data.split}

            raise ValueError(data.root)

        if isinstance(data, saev.data.datasets.ImgFolder):
            if "fish-vista" in str(data.root) and "*" in str(data.root):
                return {"name": "FistVista (Cls)", "split": "all"}

        if isinstance(data, saev.data.datasets.Imagenet):
            return {"name": "ImageNet-1K", "split": data.split}

        raise ValueError(data)

    def _():
        rows = []
        for shards_dir in shards_root.iterdir():
            md = saev.data.Metadata.load(shards_dir)
            rows.append({
                "shards": shards_dir.name,
                "family": md.family,
                "ckpt": format_ckpt(md.family, md.ckpt),
                "layers": md.layers,
                "tokens": md.content_tokens_per_example,
                **format_data_cfg(md.make_data_cfg()),
            })
            # print(shards_dir.name, md.family, md.ckpt, md.make_data_cfg())

        return pl.DataFrame(rows)

    _()
    return


if __name__ == "__main__":
    app.run()
