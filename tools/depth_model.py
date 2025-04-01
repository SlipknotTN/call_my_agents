"""
Huggingface Distill-Any-Depth-Small code from https://huggingface.co/xingyang1/Distill-Any-Depth-Small-hf
Apple DepthPro code from https://huggingface.co/apple/DepthPro
An alternative is to use the transformers.pipeline which is compatible with more depth models at the same time,
e.g. https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf and
"""

import argparse
import matplotlib
import os

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    DepthProForDepthEstimation,
    DepthProImageProcessorFast,
)


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

    if args.model_family == "distill":

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

        predicted_depth_m = (
            post_processed_output[0]["predicted_depth"].detach().cpu().numpy()
        )

        # TODO: Are the values reversed?

        # TODO: Add argument to normalize depth map or draw cmap with specific min and max values
        # Normalize depth map
        # pred_depth_normalized = (pred_depth_np - pred_depth_np.min()) / (pred_depth_np.max() - pred_depth_np.min())

        depth_colored_hwc = colorize_depth_maps(
            predicted_depth_m[None, ...], min_depth=0, max_depth=10, cmap="Spectral_r"
        ).squeeze()

    elif args.model_family == "depthpro":

        image_processor = DepthProImageProcessorFast.from_pretrained(args.hf_model_url)
        model = DepthProForDepthEstimation.from_pretrained(
            args.hf_model_url, torch_dtype=torch.float16
        ).to(args.device)

        inputs = image_processor(images=image, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        field_of_view_px = post_processed_output[0]["field_of_view"]
        focal_length_px = post_processed_output[0]["focal_length"]
        predicted_depth_m = post_processed_output[0]["predicted_depth"]

        # Normalize depth for visualization
        # depth_norm = (predicted_depth_m - predicted_depth_m.min()) / (
        #    predicted_depth_m.max() - predicted_depth_m.min()
        # )

        depth_colored_hwc = colorize_depth_maps(
            predicted_depth_m.detach().cpu().numpy()[None, ...],
            min_depth=0,
            max_depth=10,
            cmap="Spectral",
        ).squeeze()

    else:
        raise ValueError(f"Model family {args.model_family} not supported")

    # TODO: Save normalized predictions 0-1, meters 0-10, original values

    depth_colored_im = Image.fromarray(depth_colored_hwc.astype("uint8"))
    os.makedirs(args.output_dir, exist_ok=True)
    depth_colored_im.save(os.path.join(args.output_dir, "depth_colored.jpg"))


if __name__ == "__main__":
    main()
