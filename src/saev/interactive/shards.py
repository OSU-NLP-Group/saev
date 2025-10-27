import marimo

__generated_with = "0.17.2"
app = marimo.App(width="full")


@app.cell
def _():
    import pathlib

    import beartype
    import polars as pl

    import saev.data
    import saev.data.datasets
    return beartype, pathlib, pl, saev


@app.cell
def _(pathlib):
    shards_root = pathlib.Path("/fs/scratch/PAS2136/samuelstevens/saev/shards")
    return (shards_root,)


@app.cell
def _(beartype, pathlib, pl, saev, shards_root):
    BYTES_PER_GB = 1024**3
    BYTES_PER_TB = 1024**4
    BYTES_PER_FLOAT32 = 4


    @beartype.beartype
    def get_n_bytes_in_dir(shards_dpath: pathlib.Path) -> int:
        if not shards_dpath.exists():
            return 0
        if not shards_dpath.is_dir():
            return 0
        n_bytes = 0
        for path in shards_dpath.rglob("*"):
            if not path.is_file():
                continue
            n_bytes += path.stat().st_blocks * 512
        return n_bytes


    @beartype.beartype
    def bytes_to_human(n_bytes: int) -> str:
        if n_bytes >= BYTES_PER_TB:
            size = n_bytes / BYTES_PER_TB
            return f"{size:.1f} TB"
        size = n_bytes / BYTES_PER_GB
        return f"{size:.1f} GB"


    @beartype.beartype
    def make_disk_usage_metrics(shards_dpath: pathlib.Path) -> dict[str, object]:
        n_bytes = get_n_bytes_in_dir(shards_dpath)
        return {
            "disk_usage": n_bytes,
            "disk_usage_human": bytes_to_human(n_bytes),
        }


    @beartype.beartype
    def make_expected_disk_usage_metrics(md: saev.data.Metadata) -> dict[str, object]:
        n_bytes = (
            md.n_examples
            * md.tokens_per_example
            * len(md.layers)
            * md.d_model
            * BYTES_PER_FLOAT32
        )
        return {
            "expected_disk_usage": n_bytes,
            "expected_disk_usage_human": bytes_to_human(n_bytes),
        }


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


    def make_df():
        rows = []
        for shards_dpath in shards_root.iterdir():
            if not shards_dpath.is_dir():
                continue
            md = saev.data.Metadata.load(shards_dpath)
            disk_usage = make_disk_usage_metrics(shards_dpath)
            expected_disk_usage = make_expected_disk_usage_metrics(md)
            rows.append(
                {
                    "shards": shards_dpath.name,
                    "family": md.family,
                    "ckpt": format_ckpt(md.family, md.ckpt),
                    "layers": md.layers,
                    "tokens": md.content_tokens_per_example,
                    "disk": disk_usage["disk_usage_human"],
                    "expected_disk": expected_disk_usage["expected_disk_usage_human"],
                    **format_data_cfg(md.make_data_cfg()),
                }
            )
            # print(shards_dir.name, md.family, md.ckpt, md.make_data_cfg())

        return pl.DataFrame(rows)


    df = make_df()
    return (df,)


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
