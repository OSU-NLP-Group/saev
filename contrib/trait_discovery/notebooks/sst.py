# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "av==15.1.0",
#     "marimo",
#     "pillow==11.3.0",
#     "torch==2.8.0",
#     "torchvision==0.23.0",
#     "transformers==4.56.1",
# ]
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    import torchvision
    import transformers
    import transformers.video_utils

    from PIL import Image
    import av
    return torch, transformers


@app.cell
def _(torch, transformers):
    device = "cpu"
    model = transformers.Sam2VideoModel.from_pretrained("facebook/sam2.1-hiera-tiny").to(
        device, dtype=torch.bfloat16
    )
    processor = transformers.Sam2VideoProcessor.from_pretrained(
        "facebook/sam2.1-hiera-tiny"
    )
    return device, processor


@app.cell
def _(device, processor, torch, transformers):
    # For this example, we'll use the video loading utility
    video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
    video_frames, _ = transformers.video_utils.load_video(video_url)

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )
    return (inference_session,)


@app.cell
def _(inference_session, processor):
    # Add click on first frame to select object
    ann_frame_idx = 0
    ann_obj_id = 1
    points = [[[[210, 350]]]]
    labels = [[[1]]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=ann_obj_id,
        input_points=points,
        input_labels=labels,
    )
    return


@app.cell
def _(processor):
    processor
    return


@app.cell
def _(inference_session):
    inference_session
    return


@app.cell
def _():
    # Segment the object on the first frame
    # outputs = model(
    #     inference_session=inference_session,
    #     frame_idx=ann_frame_idx,
    # )
    return


@app.cell
def _():
    # video_res_masks = processor.post_process_masks(
    #     [outputs.pred_masks],
    #     original_sizes=[[inference_session.video_height, inference_session.video_width]],
    #     binarize=False,
    # )[0]
    # print(f"Segmentation shape: {video_res_masks.shape}")
    return


@app.cell
def _():
    # # Propagate through the entire video
    # video_segments = {}
    # for sam2_video_output in model.propagate_in_video_iterator(inference_session):
    #     video_res_masks = processor.post_process_masks(
    #         [sam2_video_output.pred_masks],
    #         original_sizes=[
    #             [inference_session.video_height, inference_session.video_width]
    #         ],
    #         binarize=False,
    #     )[0]
    #     video_segments[sam2_video_output.frame_idx] = video_res_masks

    # print(f"Tracked object through {len(video_segments)} frames")
    return


if __name__ == "__main__":
    app.run()
