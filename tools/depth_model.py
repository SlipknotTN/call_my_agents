"""
Huggingface Distill-Any-Depth-Small code from https://huggingface.co/xingyang1/Distill-Any-Depth-Small-hf
Apple DepthPro code from https://huggingface.co/apple/DepthPro
An alternative is to use the transformers.pipeline which is compatible with more depth models at the same time,
e.g. https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf and
"""

import argparse
import os

import matplotlib
import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          DepthProForDepthEstimation,
                          DepthProImageProcessorFast)


def colorize_depth_maps(
    depth_map,
    min_depth: float | None = None,
    max_depth: float | None = None,
    cmap: str = "Spectral",
):
    """
    Colorize depth maps

    Simplified version of colorize_depth_maps from
    https://huggingface.co/spaces/xingyang1/Distill-Any-Depth/blob/main/distillanydepth/utils/image_util.py

    Args:
        depth_map: numpy array of shape (1, H, W)
        min_depth: float, minimum depth value
        max_depth: float, maximum depth value
        cmap: str, colormap name

    Returns:
        numpy array of shape (H, W, 3)
    """
    depth = np.copy(depth_map)

    # colorize
    cm = matplotlib.colormaps[cmap]

    if min_depth != max_depth:
        depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    else:
        # Avoid 0-division
        depth = depth * 0.0

    img_colored = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored = (img_colored * 255).astype(np.uint8)

    return img_colored.squeeze()


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run Distill-Any-Depth model",
    )
    parser.add_argument(
        "--hf_model_url",
        required=False,
        type=str,
        help="HuggingFace model url",
        choices=[
            "xingyang1/Distill-Any-Depth-Small-hf",
            "xingyang1/Distill-Any-Depth-Large-hf",
            "apple/DepthPro-hf",
        ],
        default="xingyang1/Distill-Any-Depth-Small-hf",
    )
    parser.add_argument(
        "--model_family",
        required=False,
        choices=["distill", "depthpro"],
        default="depthpro",
        help="Model family the load the correct processors",
    )
    parser.add_argument(
        "--device",
        required=False,
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--image_path",
        required=False,
        type=str,
        help="Image filepath",
    )
    parser.add_argument(
        "--output_mode",
        choices=["norm", "range"],
        default="norm",
        type=str,
        help="Image output mode: norm to normalize and range to use custom min and max values (valid only for meters models)",
    )
    parser.add_argument(
        "--min_value",
        type=float,
        help="Minimum value in meters to save the output image in range mode. Values are clipped to [min_value, max_value]",
    )
    parser.add_argument(
        "--max_value",
        type=float,
        help="Maximum value in meters to save the output image in range mode. Values are clipped to [min_value, max_value]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the outputs",
    )
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    image = Image.open(args.image_path)

    if args.output_mode == "range":
        assert (
            args.min_value is not None
        ), "min_value must be set with output_mode range"
        assert (
            args.max_value is not None
        ), "max_value must be set with output_mode range"

    # TODO: Create isolated functions for relative and meters models, with inverse output arg for relative model
    if args.model_family == "distill":

        assert (
            args.output_mode == "norm"
        ), "Only norm output mode is supported for distill model, since the output is not in meters"

        image_processor = AutoImageProcessor.from_pretrained(args.hf_model_url)
        model = AutoModelForDepthEstimation.from_pretrained(args.hf_model_url)

        # prepare image for the model
        inputs = image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        # interpolate to original size and visualize the prediction
        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        predicted_depth_np_inv = (
            post_processed_output[0]["predicted_depth"].detach().cpu().numpy()
        )
        # The values are reversed. 0.0 corresponds to the farthest points
        predicted_depth_np = predicted_depth_np_inv.max() - predicted_depth_np_inv
        predicted_depth_meters_np = None
        predicted_depth_norm_np = (predicted_depth_np - predicted_depth_np.min()) / (
            predicted_depth_np.max() - predicted_depth_np.min()
        )

        depth_colored_hwc = colorize_depth_maps(
            predicted_depth_np[None, ...],
            min_depth=predicted_depth_np.min(),
            max_depth=predicted_depth_np.max(),
            cmap="Spectral",
        ).squeeze()

    elif args.model_family == "depthpro":

        image_processor = DepthProImageProcessorFast.from_pretrained(args.hf_model_url)
        model = DepthProForDepthEstimation.from_pretrained(
            args.hf_model_url, torch_dtype=torch.float16
        ).to(args.device)

        inputs = image_processor(images=image, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Post process the output using fov and focal length, it is not only a resizing of the depth map
        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        field_of_view_px = post_processed_output[0]["field_of_view"]
        focal_length_px = post_processed_output[0]["focal_length"]
        predicted_depth_meters_np = (
            post_processed_output[0]["predicted_depth"].detach().cpu().numpy()
        )
        predicted_depth_norm_np = (
            predicted_depth_meters_np - predicted_depth_meters_np.min()
        ) / (predicted_depth_meters_np.max() - predicted_depth_meters_np.min())

        depth_colored_hwc = colorize_depth_maps(
            predicted_depth_meters_np[None, ...],
            min_depth=(
                predicted_depth_meters_np.min()
                if args.output_mode == "norm"
                else args.min_value
            ),
            max_depth=(
                predicted_depth_meters_np.max()
                if args.output_mode == "norm"
                else args.max_value
            ),
            cmap="Spectral",
        ).squeeze()

    else:
        raise ValueError(f"Model family {args.model_family} not supported")

    depth_colored_im = Image.fromarray(depth_colored_hwc.astype("uint8"))
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_mode == "norm":
        output_filename = "depth_colored_norm.jpg"
    else:
        output_filename = f"depth_colored_range_{args.min_value}_{args.max_value}.jpg"
    depth_colored_im.save(os.path.join(args.output_dir, output_filename))
    np.save(os.path.join(args.output_dir, "depth_norm.npy"), predicted_depth_norm_np)
    if predicted_depth_meters_np is not None:
        np.save(
            os.path.join(args.output_dir, "depth_meters.npy"), predicted_depth_meters_np
        )


if __name__ == "__main__":
    main()
