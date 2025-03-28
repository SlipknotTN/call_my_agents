"""
Code mostly based on the official SAM2 guides:
- Images: https://github.com/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb
- Videos: https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb
- Autogenerate masks: https://github.com/facebookresearch/sam2/blob/main/notebooks/automatic_mask_generator_example.ipynb
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader, cpu
from langchain_core.tools import tool
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

np.random.seed(3)


def get_and_show_mask_image(
    mask: np.ndarray,
    random_color: bool = False,
    borders: bool = True,
    ax: plt.Axes | None = None,
) -> np.ndarray:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_overlay_with_alpha = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_overlay_with_alpha = cv2.drawContours(
            mask_overlay_with_alpha, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    if ax:
        ax.imshow(mask_overlay_with_alpha)
    return (mask_overlay_with_alpha * 255).astype(np.uint8)


def draw_and_show_points_image(
    image_canvas: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    ax: plt.Axes | None = None,
    marker_size: int = 375,
):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    if ax:
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
    for pos_point in pos_points:
        cv2.circle(
            image_canvas,
            center=pos_point,
            radius=5,
            color=(0, 255, 0),
            thickness=-1,
            lineType=-1,
        )
    for neg_point in neg_points:
        cv2.circle(
            image_canvas,
            center=neg_point,
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
            lineType=-1,
        )


def draw_and_show_box_image(
    image_canvas: np.ndarray, box: np.ndarray, ax: plt.Axes | None = None
):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    if ax:
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )
    cv2.rectangle(
        image_canvas, pt1=(x0, y0), pt2=(x0 + w, y0 + h), color=(0, 255, 0), thickness=2
    )


def get_and_show_masks_image(
    image_canvas: np.ndarray,
    masks: list | np.ndarray,
    scores: list | np.ndarray,
    point_coords: np.ndarray | None = None,
    box_coords: np.ndarray | None = None,
    input_labels: np.ndarray | None = None,
    borders: bool = True,
    plt_show: bool = False,
) -> list[np.ndarray]:
    images_with_masks = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        image_with_overlay = np.copy(image_canvas)
        if plt_show:
            plt.figure(figsize=(10, 10))
            plt.imshow(image_canvas)
            mask_overlay_with_alpha = get_and_show_mask_image(
                mask, ax=plt.gca(), borders=borders
            )
        else:
            mask_overlay_with_alpha = get_and_show_mask_image(mask, borders=borders)
        # Extract RGB and alpha channels
        mask_rgb = mask_overlay_with_alpha[:, :, :3]
        mask_alpha = mask_overlay_with_alpha[:, :, 3:4] / 255.0
        # Normalize alpha to 0-1
        image_with_overlay = (
            image_with_overlay * (1 - mask_alpha) + mask_rgb * mask_alpha
        )
        image_with_overlay = image_with_overlay.astype(np.uint8)
        if point_coords is not None:
            assert input_labels is not None
            if plt_show:
                draw_and_show_points_image(
                    image_canvas=image_with_overlay,
                    coords=point_coords,
                    labels=input_labels,
                    ax=plt.gca(),
                )
            else:
                draw_and_show_points_image(
                    image_canvas=image_with_overlay,
                    coords=point_coords,
                    labels=input_labels,
                )
        if box_coords is not None:
            # boxes
            if plt_show:
                draw_and_show_box_image(
                    image_canvas=image_with_overlay, box=box_coords, ax=plt.gca()
                )
            else:
                draw_and_show_box_image(image_canvas=image_with_overlay, box=box_coords)
        if plt_show:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
        images_with_masks.append(cv2.cvtColor(image_with_overlay, cv2.COLOR_RGB2BGR))
    return images_with_masks


def draw_and_show_mask_video_frame(
    mask: np.ndarray,
    ax: plt.Axes | None = None,
    obj_id: int | None = None,
    random_color: bool = False,
) -> np.ndarray:
    """
    Draw and show a mask on a video frame

    Args:
        mask: np.ndarray of shape (H, W)
        ax: optional plt.Axes to draw on
        obj_id: optional int to uidentify an object and reuse the same color over all frames
        random_color: bool to use a random color

    Returns:
        mask_overlay_with_alpha: np.ndarray of shape (H, W, 3)
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_overlay_with_alpha = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if ax:
        ax.imshow(mask_overlay_with_alpha)
    return (mask_overlay_with_alpha * 255).astype(np.uint8)


def save_and_show_masks_video(
    video_path: str,
    video_frames_obj_masks: dict,
    output_dir: str,
    max_frames: int | None,
    plt_show: bool = False,
) -> str:

    # Open the video again and draw masks on the frames one by one
    with open(video_path, "rb") as f:
        vr = VideoReader(f, ctx=cpu(0))
    input_video_fps = vr.get_avg_fps()
    print(f"Video frames {len(vr)}, max frames {max_frames}")
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, f"{Path(video_path).stem}_masks.mp4")

    # Get frame size from first frame
    first_frame = vr[0]
    frame_size = (first_frame.shape[1], first_frame.shape[0])

    # Initialize VideoWriter with proper fourcc code
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    vw = cv2.VideoWriter(output_video_path, fourcc, input_video_fps, frame_size)

    try:
        for i in range(len(vr)):
            if max_frames is not None and i >= max_frames:
                break
            # the video reader will handle seeking and skipping in the most efficient manner
            frame = vr[i].cpu().numpy()
            if plt_show:
                plt.title(f"frame {i}")
                plt.imshow(frame)

            frame_with_overlay = frame.copy()
            if i in video_frames_obj_masks.keys():
                for out_obj_id, out_mask in video_frames_obj_masks[i].items():
                    if plt_show:
                        mask_overlay_with_alpha = draw_and_show_mask_video_frame(
                            out_mask, plt.gca(), obj_id=out_obj_id
                        )
                        if i % 10 == 0:
                            # show every 10 frames
                            plt.show()
                    else:
                        mask_overlay_with_alpha = draw_and_show_mask_video_frame(
                            out_mask, obj_id=out_obj_id
                        )
                    # Extract RGB and alpha channels
                    mask_rgb = mask_overlay_with_alpha[:, :, :3]
                    mask_alpha = mask_overlay_with_alpha[:, :, 3:4] / 255.0
                    # Normalize alpha to 0-1
                    frame_with_overlay = (
                        frame_with_overlay * (1 - mask_alpha) + mask_rgb * mask_alpha
                    )
                    frame_with_overlay = frame_with_overlay.astype(np.uint8)
                vw.write(cv2.cvtColor(frame_with_overlay, cv2.COLOR_RGB2BGR))
            else:
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()
    return output_video_path


@tool
def predict_image_masks(
    image_path: str,
    hf_model_url: str,
    prompt_points: list[list[float]] | None = None,
    prompt_labels: list[int] | None = None,
    prompt_box: list[float] | None = None,
    multimask_output: bool = False,
) -> dict:
    """
    Predict masks for an image using a SAM2 image model. Pydantic compatible, no need numpy arrays in the interface.

    Args:
        image_path: str, path to the image to predict masks for
        hf_model_url: str, Hugging Face model url. E.g. facebook/sam2-hiera-large or facebook/sam2-hiera-small
        prompt_points: list of [x, y] coordinates, optional points to prompt the model
        prompt_labels: list of integers (0 or 1), optional labels to prompt the model
        prompt_box: list of [x1, y1, x2, y2], optional box to prompt the model
        multimask_output: bool, optional, whether to return multiple masks

    Returns:
        dict with masks, scores, and mask_logits encoded as lists
    """
    im_predictor = SAM2ImagePredictor.from_pretrained(hf_model_url)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # Convert lists to numpy arrays if provided
    np_prompt_points = np.array(prompt_points) if prompt_points is not None else None
    np_prompt_labels = np.array(prompt_labels) if prompt_labels is not None else None
    np_prompt_box = np.array(prompt_box) if prompt_box is not None else None

    masks, scores, masks_logits = predict_image_masks_from_model(
        im_predictor=im_predictor,
        image=image,
        prompt_points=np_prompt_points,
        prompt_labels=np_prompt_labels,
        prompt_box=np_prompt_box,
        multimask_output=multimask_output,
    )

    # Convert numpy arrays to lists for Pydantic compatibility
    return {
        "masks": masks.tolist(),
        "scores": scores.tolist(),
        "masks_logits": masks_logits.tolist(),
    }


def predict_image_masks_from_model(
    im_predictor: SAM2ImagePredictor,
    image: np.ndarray,
    prompt_points: np.ndarray | None = None,
    prompt_labels: np.ndarray | None = None,
    prompt_box: np.ndarray | None = None,
    multimask_output: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict masks for an image using an initialized SAM2ImagePredictor

    Args:
        im_predictor: SAM2ImagePredictor
        image: np.ndarray in RGB format uint8
        prompt_points: np.ndarray of shape (N, 2)
        prompt_labels: np.ndarray of shape (N,)
        prompt_box: np.ndarray of shape (4,)
        multimask_output: returns 3 masks if True, otherwise only the best mask

    Returns:
        masks: np.ndarray of shape (H, W)
        scores: list of float
        masks_logits: np.ndarray of shape (H, W)
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Set embeddings
        im_predictor.set_image(image)
        # Run prediction
        masks, scores, masks_logits = im_predictor.predict(
            point_coords=prompt_points,
            point_labels=prompt_labels,
            box=prompt_box,
            normalize_coords=True,
            multimask_output=multimask_output,
        )
        return masks, scores, masks_logits


@tool
def predict_video_masks(
    video_path: str,
    hf_model_url: str,
    prompt_points: list[list[float]],
    prompt_labels: list[int],
    prompt_box: list[float] | None = None,
    max_frames: int | None = None,
) -> dict:
    """
    Predict masks for a video using a SAM2 video model. Pydantic compatible, no need numpy arrays in the interface.

    Args:
        video_path: str, path to the video to predict masks for
        hf_model_url: str, Hugging Face model url. E.g. facebook/sam2-hiera-large or facebook/sam2-hiera-small
        prompt_points: list of lists, each inner list contains [x, y] coordinates
        prompt_labels: list of integers (0 for negative, 1 for positive points)
        prompt_box: list of 4 values [x1, y1, x2, y2] defining a bounding box
        max_frames: int, optional, maximum number of frames to process

    Returns:
        video_frames_obj_masks: dict, keys are frame indices, values are dictionaries of object ids and their masks
    """
    video_predictor = SAM2VideoPredictor.from_pretrained(hf_model_url)

    # Convert lists to numpy arrays for processing
    np_prompt_points = np.array(prompt_points, dtype=np.float32)
    np_prompt_labels = np.array(prompt_labels, dtype=np.int32)
    np_prompt_box = np.array(prompt_box, dtype=np.float32) if prompt_box else None

    result = predict_video_masks_from_model(
        video_predictor=video_predictor,
        video_path=video_path,
        start_frame_idx=0,
        prompt_points=np_prompt_points,
        prompt_labels=np_prompt_labels,
        prompt_box=np_prompt_box,
        max_frames=max_frames,
    )

    # Convert any numpy arrays in the result to lists for Pydantic compatibility
    pydantic_result = {}
    for frame_idx, frame_data in result.items():
        pydantic_result[frame_idx] = {}
        for obj_id, mask in frame_data.items():
            if isinstance(mask, np.ndarray):
                pydantic_result[frame_idx][obj_id] = mask.tolist()
            else:
                pydantic_result[frame_idx][obj_id] = mask

    return pydantic_result


def predict_video_masks_from_model(
    video_predictor: SAM2VideoPredictor,
    video_path: str,
    start_frame_idx: int,
    prompt_points: np.ndarray,
    prompt_labels: np.ndarray,
    prompt_box: np.ndarray | None = None,
    max_frames: int | None = None,
) -> dict:
    """
    Predict masks for a video using an initialized SAM2VideoPredictor

    Args:
        video_predictor: SAM2VideoPredictor
        video_path: str, path to the video to predict masks for
        start_frame_idx: int, index of the first frame to process
        prompt_points: np.ndarray of shape (N, 2)
        prompt_labels: np.ndarray of shape (N,)
        prompt_box: np.ndarray of shape (4,)
        max_frames: int, optional, maximum number of frames to process

    Returns:
        video_frames_obj_masks: dict, keys are frame indices, values are dictionaries of object ids and their masks
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Init inference state, call predictor.reset_state(inference_state) to restart
        state = video_predictor.init_state(video_path, offload_video_to_cpu=True)

        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks_logits = video_predictor.add_new_points_or_box(
            state,
            frame_idx=start_frame_idx,
            obj_id=0,
            points=prompt_points,
            labels=prompt_labels,
            box=prompt_box,
        )

        # run propagation throughout the video without updating the prompt and collect the results in a dict
        video_frames_obj_masks = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in video_predictor.propagate_in_video(state):
            if max_frames is not None and out_frame_idx >= max_frames:
                break
            video_frames_obj_masks[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_frames_obj_masks


def do_parsing():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run SAM 2 model",
    )
    parser.add_argument(
        "--hf_model_url",
        required=False,
        type=str,
        help="Hugging Face model url",
        default="facebook/sam2-hiera-large",
    )
    parser.add_argument(
        "--image_path",
        required=False,
        type=str,
        help="Image filepath",
    )
    parser.add_argument(
        "--video_path",
        required=False,
        type=str,
        help="Image filepath",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        required=False,
        help="Maximum number of frames to process (start from 0 excluding start_frame_idx)",
    )
    parser.add_argument(
        "--show_best",
        action="store_true",
        help="Show the mask with the highest score only",
    )
    parser.add_argument(
        "--show_with_mpl",
        action="store_true",
        help="Show the results with matplotlib interactively",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save images with masks",
    )
    return parser.parse_args()


def main():
    args = do_parsing()
    print(args)

    if args.image_path:
        assert args.max_frames is None, "Cannot set both max_frames and image_path"

    assert (
        int(args.image_path is not None) + int(args.video_path is not None) == 1
    ), f"Exactly one of image_path or video_path must be set"

    if args.image_path:
        im_predictor = SAM2ImagePredictor.from_pretrained(args.hf_model_url)

        image = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)

        # Test positive and negative points valid for cat.jpg image
        prompt_points = np.array([[250, 180], [100, 100]])
        # Prompt label is necessary to assign positive (1) and negative (0) points
        prompt_labels = np.array([1, 0])
        # Test box prompt valid for cat.jpg image
        prompt_box = np.array([170, 50, 340, 310])

        masks, scores, masks_logits = predict_image_masks_from_model(
            im_predictor,
            image,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            prompt_box=prompt_box,
            multimask_output=not args.show_best,
        )

        images_with_masks = get_and_show_masks_image(
            image_canvas=image,
            masks=masks,
            scores=scores,
            point_coords=prompt_points,
            input_labels=prompt_labels,
            box_coords=prompt_box,
            plt_show=args.show_with_mpl,
        )
        assert len(images_with_masks) > 0, "No images with masks to draw"
        os.makedirs(args.output_dir, exist_ok=True)
        for idx, image_with_masks in enumerate(images_with_masks):
            cv2.imwrite(
                os.path.join(
                    args.output_dir,
                    f"{Path(args.image_path).stem}_{idx}_score_{scores[idx]}.jpg",
                ),
                image_with_masks,
            )

    else:
        video_predictor = SAM2VideoPredictor.from_pretrained(args.hf_model_url)

        # Test a point valid for people_walking_2.mp4 video
        prompt_points = np.array([[712, 578]])
        # Prompt label is necessary to assign positive (1) and negative (0) points
        prompt_labels = np.array([1])
        # No box prompt
        prompt_box = None
        # Skip the first 4 frames
        start_frame_idx = 4
        video_frames_obj_masks = predict_video_masks_from_model(
            video_predictor,
            video_path=args.video_path,
            start_frame_idx=start_frame_idx,
            prompt_points=prompt_points,
            prompt_labels=prompt_labels,
            prompt_box=prompt_box,
            max_frames=args.max_frames,
        )

        assert len(video_frames_obj_masks) > 0, "No video frames with masks to draw"
        save_and_show_masks_video(
            video_path=args.video_path,
            video_frames_obj_masks=video_frames_obj_masks,
            output_dir=args.output_dir,
            max_frames=args.max_frames,
            plt_show=args.show_with_mpl,
        )


if __name__ == "__main__":
    main()
