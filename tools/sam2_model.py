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
import numpy as np
import torch
import matplotlib.pyplot as plt

from decord import VideoReader
from decord import cpu, gpu
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

np.random.seed(3)


def get_and_show_mask(mask, random_color=False, borders=True, ax=None) -> np.ndarray:
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_color_with_alpha = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_color_with_alpha = cv2.drawContours(
            mask_color_with_alpha, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    if ax:
        ax.imshow(mask_color_with_alpha)
    return (mask_color_with_alpha * 255).astype(np.uint8)


def get_and_show_points(image_width: int, image_height: int, coords: np.ndarray, labels: np.ndarray, ax: plt.axes, marker_size: int = 375) -> np.ndarray:
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    points_overlay = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
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
        cv2.circle(points_overlay, center=pos_point, radius=10, color=(0, 255, 0), thickness=2, lineType=-1)
    for neg_point in neg_points:
        cv2.circle(points_overlay, center=neg_point, radius=10, color=(0, 0, 255), thickness=2, lineType=-1)
    return points_overlay

def get_and_show_box(image_width: int, image_height: int, box, ax) -> np.ndarray:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    box_overlay = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
    if ax:
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
        )
    cv2.rectangle(box_overlay, pt1=(x0, y0), pt2=(x0 + w, y0 + h), color=(0, 255, 0), thickness=2)
    return box_overlay

def get_and_show_masks(
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
            mask_overlay_with_alpha = get_and_show_mask(mask, ax=plt.gca(), borders=borders)
        else:
            mask_overlay_with_alpha = get_and_show_mask(mask, borders=borders)
        image_with_overlay = cv2.addWeighted(image_with_overlay, 1.0, mask_overlay_with_alpha[:,:,:3], 0.6, 0)
        if point_coords is not None:
            assert input_labels is not None
            if plt_show:
                points_overlay = get_and_show_points(image_width=mask.shape[1], image_height=mask.shape[0],
                                                     coords=point_coords, labels=input_labels, ax=plt.gca())
            else:
                points_overlay = get_and_show_points(image_width=mask.shape[1], image_height=mask.shape[0],
                                                     coords=point_coords, labels=input_labels)
            image_with_overlay = cv2.addWeighted(image_with_overlay, 1.0, points_overlay, 1.0, 0)
        if box_coords is not None:
            # boxes
            if plt_show:
                box_overlay = get_and_show_box(image_width=mask.shape[1], image_height=mask.shape[0], box=box_coords, ax=plt.gca())
            else:
                box_overlay = get_and_show_box(image_width=mask.shape[1], image_height=mask.shape[0], box=box_coords)
            image_with_overlay = cv2.addWeighted(image_with_overlay, 1.0, box_overlay, 1.0, 0)
        if plt_show:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis("off")
            plt.show()
        images_with_masks.append(cv2.cvtColor(image_with_overlay, cv2.COLOR_RGB2BGR))
    return images_with_masks


def show_mask_video(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


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
        "--show_best",
        action="store_true",
        help="Show the mask with the highest score only",
    )
    parser.add_argument(
        "--show_with_mpl",
        action="store_true",
        help="Show the results with matplotlib interactively"
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

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Set embeddings
            im_predictor.set_image(image)
            # Run prediction
            masks, scores, masks_logits = im_predictor.predict(
                point_coords=prompt_points,
                point_labels=prompt_labels,
                box=prompt_box,
                normalize_coords=True,
                multimask_output=not args.show_best,
            )

        images_with_masks = get_and_show_masks(
            image_canvas=image,
            masks=masks,
            scores=scores,
            point_coords=prompt_points,
            input_labels=prompt_labels,
            box_coords=prompt_box,
            plt_show=args.show_with_mpl
        )
        assert len(images_with_masks) > 0, "No images with masks to draw"
        os.makedirs(args.output_dir, exist_ok=True)
        for idx, image_with_masks in enumerate(images_with_masks):
            cv2.imwrite(
                os.path.join(
                    args.output_dir,
                    f"{Path(args.image_path).stem}_{idx}_score_{scores[idx]}.jpg",
                ),
                image_with_masks
            )

    else:
        predictor = SAM2VideoPredictor.from_pretrained(args.hf_model_url)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Init inference state, call predictor.reset_state(inference_state) to restart
            state = predictor.init_state(args.video_path, offload_video_to_cpu=True)

            # Test a point valid for people_walking_2.mp4 video
            prompt_points = np.array([[712, 578]])
            # Prompt label is necessary to assign positive (1) and negative (0) points
            prompt_labels = np.array([1])

            start_frame_idx = 4

            # add new prompts and instantly get the output on the same frame
            frame_idx, object_ids, masks_logits = predictor.add_new_points_or_box(
                state,
                frame_idx=start_frame_idx,
                obj_id=0,
                points=prompt_points,
                labels=prompt_labels,
            )

            max_frames = 50

            # run propagation throughout the video without updating the prompt and collect the results in a dict
            video_segments = (
                {}
            )  # video_segments contains the per-frame segmentation results
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in predictor.propagate_in_video(state):
                if out_frame_idx >= max_frames:
                    break
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            # vis_frame_stride = 30
            plt.close("all")

            # Open the video again and draw masks on the frames
            # TODO: Create an output video
            with open(args.video_path, "rb") as f:
                vr = VideoReader(f, ctx=cpu(0))
            print("video frames:", len(vr))
            # The simplest way is to directly access frames
            for i in range(len(vr)):
                if i >= max_frames:
                    break
                # the video reader will handle seeking and skipping in the most efficient manner
                frame = vr[i]
                plt.title(f"frame {i}")
                # TODO: This assumes the frames are already saved as images
                # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
                plt.imshow(frame)
                if i in video_segments.keys():
                    for out_obj_id, out_mask in video_segments[i].items():
                        mask_image = show_mask_video(
                            out_mask, plt.gca(), obj_id=out_obj_id
                        )
                        plt.show()
                        cv2.imwrite(
                            f"./data/{i}_{out_obj_id}.jpg", mask_image[:, :, :3] * 255
                        )


if __name__ == "__main__":
    main()
