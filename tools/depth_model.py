"""
Huggingface Distill-Any-Depth-Small code from https://huggingface.co/xingyang1/Distill-Any-Depth-Small-hf
An alternative is to use the transformers.pipeline which is compatible with more depth models at the same time,
e.g. https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf and
"""

import argparse
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

        predicted_depth = post_processed_output[0]["predicted_depth"]

    elif args.model_family == "depthpro":
        # FIXME: OOM, try another one like depth anything 2
        image_processor = DepthProImageProcessorFast.from_pretrained(args.hf_model_url)
        model = DepthProForDepthEstimation.from_pretrained(args.hf_model_url).to(
            args.device
        )

        inputs = image_processor(images=image, return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)

        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        field_of_view = post_processed_output[0]["field_of_view"]
        focal_length = post_processed_output[0]["focal_length"]
        predicted_depth = post_processed_output[0]["predicted_depth"]

    else:
        raise ValueError(f"Model family {args.model_family} not supported")

    # Normalize depth for visualization
    depth = (predicted_depth - predicted_depth.min()) / (
        predicted_depth.max() - predicted_depth.min()
    )
    depth = depth.detach().cpu().numpy() * 255
    depth = Image.fromarray(depth.astype("uint8"))
    os.makedirs(args.output_dir, exist_ok=True)
    depth.save(os.path.join(args.output_dir, "depth_normalized.jpg"))

    # TODO: Save the original predictions as well, are the values in metrics


if __name__ == "__main__":
    main()
