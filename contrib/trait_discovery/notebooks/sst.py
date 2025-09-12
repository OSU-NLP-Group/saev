# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype==0.21.0",
#     "duckdb==1.3.2",
#     "jaxtyping==0.3.2",
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.3.3",
#     "pillow==11.3.0",
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "transformers==4.56.1",
#     "polars==1.33.1",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import pathlib

    import beartype
    import duckdb
    import marimo as mo
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import torch
    import transformers
    from jaxtyping import Bool, Float, jaxtyped
    from PIL import Image
    from torch import Tensor
    return (
        Bool,
        Float,
        Image,
        Tensor,
        beartype,
        duckdb,
        jaxtyped,
        mo,
        np,
        patches,
        pathlib,
        pl,
        plt,
        torch,
        transformers,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Static Segmentation Through Tracking (SST) Interactive Notebook

    An interactive marimo notebook that doesn't need any Python programming to run SST using SAM 2.1 on high-resolution images.

    [SST Paper](https://arxiv.org/abs/2501.06749)
    """
    )
    return


@app.cell
def _(mo):
    csv_path_input = mo.ui.text(
        value="",
        label="Path to master CSV file:",
        placeholder="/path/to/master.csv",
        full_width=True,
    )

    mo.vstack([mo.md("## Step 1: Load Data"), csv_path_input])
    return (csv_path_input,)


@app.cell
def _(csv_path_input, duckdb, mo, pathlib, pl):
    def _():
        csv_path = pathlib.Path(csv_path_input.value) if csv_path_input.value else None

        if not csv_path_input.value:
            return None, None, mo.md("Enter a path to the master CSV file to begin")

        if not csv_path.exists():
            return None, None, mo.md(f"❌ File not found: {csv_path_input.value}")

        conn = duckdb.connect()
        master_df = pl.read_csv(csv_path, infer_schema_length=None)
        # Register with DuckDB for SQL queries
        conn.register("master_df", master_df)

        status_md = mo.vstack(
            [
                mo.md(
                    f"✅ Loaded CSV with {len(master_df)} rows and {len(master_df.columns)} columns"
                ),
                mo.ui.table(master_df.head(10)),
            ]
        )
        return conn, master_df, status_md


    conn, master_df, _status_md = _()
    csv_path = pathlib.Path(csv_path_input.value) if csv_path_input.value else None
    _status_md
    return conn, csv_path


@app.cell
def _(csv_path, mo):
    def _():
        if not (csv_path and csv_path.exists()):
            return None, None, None, mo.md("Load a CSV file first")

        filter_query_input = mo.ui.text_area(
            value="SELECT * FROM master_df",
            label="SQL query to filter rows:",
            placeholder="SELECT * FROM master_df WHERE condition",
            full_width=True,
        )

        group_by_input = mo.ui.text(
            value="",
            label="Column(s) to group images by (comma-separated):",
            placeholder="experiment_id, subject_id",
            full_width=True,
        )

        image_path_input = mo.ui.text(
            value="",
            label="Column name or SQL expression for image paths:",
            placeholder="image_path or CONCAT(base_dir, '/', filename)",
            full_width=True,
        )

        # Return empty status since we'll display in next cell
        return filter_query_input, group_by_input, image_path_input, mo.md("")


    filter_query_input, group_by_input, image_path_input, _status_md = _()
    _status_md
    return filter_query_input, group_by_input, image_path_input


@app.cell
def _(filter_query_input, group_by_input, image_path_input, mo):
    mo.vstack(
        [
            mo.md("## Step 2: Configure Data Processing"),
            filter_query_input,
            group_by_input,
            image_path_input,
        ]
    )
    return


@app.cell
def _(
    conn,
    csv_path,
    filter_query_input,
    group_by_input,
    image_path_input,
    mo,
    pl,
):
    def _():
        if not (csv_path and csv_path.exists()):
            return {}, None, mo.md("Load a CSV file first")

        if not (
            filter_query_input
            and filter_query_input.value
            and image_path_input
            and image_path_input.value
        ):
            return {}, None, mo.md("Configure the data processing settings above")

        try:
            filtered_df = conn.execute(filter_query_input.value).pl()

            if image_path_input.value not in filtered_df.columns:
                path_query = (
                    f"SELECT *, {image_path_input.value} as image_path FROM filtered_df"
                )
                filtered_df = conn.execute(path_query).pl()
            else:
                filtered_df = filtered_df.with_columns(
                    pl.col(image_path_input.value).alias("image_path")
                )

            if group_by_input.value:
                group_cols = [col.strip() for col in group_by_input.value.split(",")]
                # Group by columns and aggregate image paths
                grouped = filtered_df.group_by(group_cols).agg(pl.col("image_path"))
                # Convert to dictionary format
                image_groups = {}
                for row in grouped.iter_rows(named=True):
                    key = tuple(row[col] for col in group_cols)
                    image_groups[key] = row["image_path"]

                status_md = mo.vstack(
                    [
                        mo.md(f"✅ Created {len(image_groups)} image groups"),
                        mo.md(
                            f"Group sizes: {[len(imgs) for imgs in list(image_groups.values())[:10]]}..."
                        ),
                    ]
                )
            else:
                image_groups = {
                    i: [path] for i, path in enumerate(filtered_df["image_path"])
                }
                status_md = mo.md(f"✅ Processing {len(image_groups)} individual images")

            return image_groups, filtered_df, status_md

        except Exception as e:
            return {}, None, mo.md(f"❌ Error processing data: {str(e)}")


    image_groups, filtered_df, _status_md = _()
    _status_md
    return (image_groups,)


@app.cell
def _(Image, beartype, pathlib):
    @beartype.beartype
    class Frame:
        def __init__(self, path: pathlib.Path):
            self.path = path
            self._img = None

        @property
        def img(self):
            if self._img is None:
                self._img = Image.open(self.path)
            return self._img
    return (Frame,)


@app.cell
def _(Bool, Float, Frame, Image, Tensor, beartype, jaxtyped, np, patches, plt):
    @jaxtyped(typechecker=beartype.beartype)
    def display_frame(
        frame: Frame,
        *,
        scale: float = 0.2,
        masks: Bool[Tensor, "n_masks height width"] | None = None,
        points: Float[Tensor, "n_points 2"] | None = None,
        boxes: Float[Tensor, "n_boxes 4"] | None = None,
    ):
        img = frame.img
        orig_width, orig_height = img.size
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img_scaled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        fig, ax = plt.subplots(figsize=(10, 10 * new_height / new_width))
        ax.imshow(img_scaled)
        ax.axis("off")

        if masks is not None:
            masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else masks
            for i, mask in enumerate(masks_np):
                mask_scaled = Image.fromarray(mask.astype(np.uint8) * 255)
                mask_scaled = mask_scaled.resize(
                    (new_width, new_height), Image.Resampling.NEAREST
                )
                mask_scaled = np.array(mask_scaled) > 127

                color = plt.cm.tab10(i % 10)
                overlay = np.zeros((new_height, new_width, 4))
                overlay[:, :, :3] = color[:3]
                overlay[:, :, 3] = mask_scaled * 0.7
                ax.imshow(overlay)

        if points is not None:
            points_np = points.cpu().numpy() if hasattr(points, "cpu") else points
            points_scaled = points_np * scale
            ax.scatter(
                points_scaled[:, 0],
                points_scaled[:, 1],
                c="lime",
                s=100,
                marker="x",
                linewidths=2,
            )

        if boxes is not None:
            boxes_np = boxes.cpu().numpy() if hasattr(boxes, "cpu") else boxes
            boxes_scaled = boxes_np * scale
            for box in boxes_scaled:
                width = box[2] - box[0]
                height = box[3] - box[1]
                rect = patches.Rectangle(
                    (box[0], box[1]),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="cyan",
                    facecolor="none",
                )
                ax.add_patch(rect)

        plt.tight_layout()
        return fig
    return (display_frame,)


@app.cell
def _(image_groups, mo):
    def _():
        if not image_groups:
            return None, mo.md(
                "No image groups available. Configure data processing first."
            )

        group_keys = list(image_groups.keys())
        group_selector = mo.ui.dropdown(
            options={str(k): k for k in group_keys[:100]},
            label="Select image group:",
            full_width=True,
        )

        status_md = mo.vstack([mo.md("## Step 3: Select Image Group"), group_selector])
        return group_selector, status_md


    group_selector, _status_md = _()
    _status_md
    return (group_selector,)


@app.cell
def _(Frame, group_selector, image_groups, mo, pathlib):
    def _():
        if not (image_groups and group_selector and group_selector.value is not None):
            return [], {}, None, mo.md("")

        selected_group = group_selector.value
        image_paths = image_groups[selected_group]

        frames = [Frame(pathlib.Path(path)) for path in image_paths]

        first_idx = 0
        middle_idx = len(frames) // 2

        key_frames = {
            "first": (first_idx, frames[first_idx]),
            "middle": (middle_idx, frames[middle_idx]),
        }

        status_md = mo.vstack(
            [
                mo.md(f"✅ Group has {len(frames)} images"),
                mo.md(
                    f"Processing frame {first_idx} (first) and frame {middle_idx} (middle)"
                ),
            ]
        )
        return frames, key_frames, selected_group, status_md


    frames, key_frames, selected_group, _status_md = _()
    _status_md
    return frames, key_frames


@app.cell
def _(mo, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mo.md(f"Using device: **{device}**")
    return (device,)


@app.cell
def _(key_frames, mo):
    def _():
        if not key_frames:
            return None, mo.md("Select a group first to generate masks")

        generate_masks_button = mo.ui.run_button(label="Generate Masks with SAM 2.1")
        status_md = mo.vstack(
            [
                mo.md("## Step 4: Generate Initial Masks"),
                generate_masks_button,
            ]
        )
        return generate_masks_button, status_md


    generate_masks_button, _status_md = _()
    _status_md
    return (generate_masks_button,)


@app.cell
def _(device, generate_masks_button, key_frames, mo, transformers):
    def _():
        if not (generate_masks_button and generate_masks_button.value and key_frames):
            return {}, None, mo.md("")

        status_messages = ["🔄 Loading SAM 2.1 model..."]

        generator = transformers.pipeline(
            "mask-generation", model="facebook/sam2.1-hiera-tiny", device=device
        )

        frame_masks = {}
        for fname, (_, frame) in key_frames.items():
            status_messages.append(f"🔄 Generating masks for {fname} frame...")
            pipeline_outputs = generator(frame.img, points_per_batch=64)
            frame_masks[fname] = pipeline_outputs

        status_messages.append("✅ Mask generation complete!")
        status_md = mo.md("\n\n".join(status_messages))
        return frame_masks, generator, status_md


    frame_masks, generator, _status_md = _()
    _status_md
    return (frame_masks,)


@app.cell
def _(frame_masks, mo):
    mask_selectors = {}
    obj_id_inputs = {}
    mo.stop(not frame_masks)

    for fkey in frame_masks.keys():
        # Create checkboxes and object ID inputs for all masks
        n_masks = len(frame_masks[fkey]["masks"])

        # Create pairs of checkbox and number input for each mask
        mask_controls = []
        obj_id_controls = []
        for i in range(n_masks):
            mask_controls.append(mo.ui.checkbox(label=f"Mask {i}", value=False))
            obj_id_controls.append(
                mo.ui.number(
                    value=i, start=0, stop=n_masks * 2, label="Object", full_width=False
                )
            )

        mask_selectors[fkey] = mo.ui.array(mask_controls)
        obj_id_inputs[fkey] = mo.ui.array(obj_id_controls)

    mask_selectors = mo.ui.dictionary(mask_selectors)
    obj_id_inputs = mo.ui.dictionary(obj_id_inputs)
    return mask_selectors, obj_id_inputs


@app.cell
def _(mo):
    tabs = mo.ui.tabs({"first": mo.md(""), "middle": mo.md("")})
    return (tabs,)


@app.cell
def _(tabs):
    tabs
    return


@app.cell
def _(grid, mask_selectors, tabs):
    grid(mask_selectors[tabs.value], n_rows=4, n_cols=6)
    return


@app.cell
def _(grid, obj_id_inputs, tabs):
    grid(obj_id_inputs[tabs.value], n_rows=4, n_cols=6)
    return


@app.cell
def _(display_frame, frame_masks, key_frames, mask_selectors, mo, tabs, torch):
    mo.stop(not frame_masks)

    _, frame = key_frames[tabs.value]

    # Get selected mask indices
    selected_indices = [i for i, cb in enumerate(mask_selectors[tabs.value].value) if cb]

    # Create visualization with selected masks
    if selected_indices:
        selected_mask_list = [frame_masks[tabs.value]["masks"][i] for i in selected_indices]
        masks_tensor = torch.stack([torch.tensor(m) for m in selected_mask_list])
        fig = display_frame(frame, scale=0.05, masks=masks_tensor)
    else:
        # Show image without masks if none selected
        fig = display_frame(frame, scale=0.05, masks=None)

    fig
    return


@app.cell
def _(frame_masks, mask_selectors, mo, obj_id_inputs):
    mo.stop(not frame_masks)


    # Collect mask-to-object mappings for each frame
    object_mappings = {}
    selected_masks = {}

    for frame_name, _ in mask_selectors.items():
        frame_mappings = []
        frame_selected = []

        for m_idx, (cb, obj_id) in enumerate(zip(mask_selectors.value[frame_name], obj_id_inputs[frame_name])):
            if cb and obj_id.value > 0:
                frame_mappings.append((m_idx, obj_id.value))
                frame_selected.append(m_idx)

        if frame_mappings:
            object_mappings[frame_name] = frame_mappings
            selected_masks[frame_name] = frame_selected


    linkages = {}
    for frame_name, mappings in object_mappings.items():
        for mask_idx, obj_id in mappings:
            if obj_id not in linkages:
                linkages[obj_id] = {}
            if frame_name not in linkages[obj_id]:
                linkages[obj_id][frame_name] = []
            linkages[obj_id][frame_name].append(mask_idx)

    # Create summary table
    table_data = []
    for obj_id in sorted(linkages.keys()):
        first_masks = linkages[obj_id].get("first", [])
        middle_masks = linkages[obj_id].get("middle", [])
        table_data.append(
            {
                "Object ID": obj_id,
                "First Frame Masks": ", ".join(map(str, first_masks))
                if first_masks
                else "-",
                "Middle Frame Masks": ", ".join(map(str, middle_masks))
                if middle_masks
                else "-",
            }
        )

    mo.vstack(
        [
            mo.md("### Object Linkages"),
            mo.ui.table(table_data) if table_data else mo.md("No linked objects yet"),
        ]
    )
    return object_mappings, selected_masks


@app.cell
def _(display_frame, frame_masks, key_frames, mo, selected_masks, torch):
    def _():
        if not (selected_masks and frame_masks):
            return mo.md("")

        selected_displays = []
        for f_name, indices in selected_masks.items():
            if f_name in frame_masks and f_name in key_frames:
                _, selected_frame = key_frames[f_name]
                selected_mask_data = frame_masks[f_name]

                selected_mask_list = [selected_mask_data["masks"][i] for i in indices]
                if selected_mask_list:
                    sel_masks_tensor = torch.stack(
                        [torch.tensor(m) for m in selected_mask_list]
                    )

                    sel_fig = display_frame(
                        selected_frame, scale=0.2, masks=sel_masks_tensor
                    )
                    selected_displays.append(
                        mo.vstack(
                            [
                                mo.md(
                                    f"### Selected Masks for {f_name.capitalize()} Frame"
                                ),
                                sel_fig,
                            ]
                        )
                    )

        if selected_displays:
            return mo.vstack(selected_displays)
        return mo.md("")


    _status_md = _()
    _status_md
    return


@app.cell
def _(frames, mo, object_mappings):
    def _():
        if not (object_mappings and frames and len(frames) > 2):
            if frames:
                return None, mo.md("Assign Object IDs to masks first (Object ID > 0)")
            else:
                return None, mo.md("")

        # Count unique objects
        unique_objects = set()
        for frame_mappings in object_mappings.values():
            for _, obj_id in frame_mappings:
                unique_objects.add(obj_id)

        track_button = mo.ui.run_button(label="Run SST Tracking on All Frames")
        status_md = mo.vstack(
            [
                mo.md("## Step 6: Track Through All Frames"),
                mo.md(
                    f"Ready to track {len(unique_objects)} objects through {len(frames)} frames"
                ),
                track_button,
            ]
        )
        return track_button, status_md


    track_button, _status_md = _()
    _status_md
    return (track_button,)


@app.cell
def _(device, frames, mo, torch, track_button, transformers):
    def _():
        if not (track_button and track_button.value and frames):
            return None, None, mo.md("")

        status_messages = ["🔄 Initializing SAM 2 Video model for tracking..."]

        model = transformers.Sam2VideoModel.from_pretrained(
            "facebook/sam2.1-hiera-tiny"
        ).to(device, dtype=torch.bfloat16)

        processor = transformers.Sam2VideoProcessor.from_pretrained(
            "facebook/sam2.1-hiera-tiny"
        )

        status_messages.append("✅ Video model loaded")
        status_md = mo.md("\n\n".join(status_messages))
        return model, processor, status_md


    model, processor, _status_md = _()
    _status_md
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Instructions

    1. **Load your data**: Provide a path to a CSV file containing image metadata
    2. **Configure processing**: Write SQL queries to filter and group your images
    3. **Select a group**: Choose which group of images to process
    4. **Generate masks**: Use SAM 2.1 to generate candidate masks for key frames
    5. **Select masks**: Choose which masks to track through the sequence
    6. **Run tracking**: Apply SST to propagate masks through all frames
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ---

    ### Notes
    - Images are automatically downscaled for faster interactive display
    - The notebook processes the first and middle frames of each group
    - SAM 2.1 Hiera-Tiny model is used for efficiency on high-res images
    """
    )
    return


@app.cell
def _(mo):
    def grid(things, *, n_cols: int = 4, n_rows=4):
        n_things = min(len(things), n_cols * n_rows)

        cols = []
        for col_start in range(0, n_things, n_rows):
            col_end = min(col_start + n_rows, n_things)
            col_things = [things[i] for i in range(col_start, col_end)]
            if col_things:  # Only add non-empty rows
                cols.append(mo.vstack(col_things))

        return mo.hstack(cols)
    return (grid,)


if __name__ == "__main__":
    app.run()
