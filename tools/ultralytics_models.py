import argparse
import os

import ultralytics
from ultralytics import YOLO


def do_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="./data/cat.jpg")
    parser.add_argument(
        "--model_path",
        type=str,
        default="yolo11s-seg.pt",
        choices=[
            "yolo11s-seg.pt",
            "yolo11s-pose.pt",
            "yolo11m-seg.pt",
            "yolo11m-pose.pt",
        ],
        help="Pretrained YOLO model",
    )
    parser.add_argument(
        "--show_pred", action="store_true", help="Show predicted bboxes"
    )
    parser.add_argument("--output_path", type=str, required=False, help="Output path")
    return parser.parse_args()


def predict_bboxes_and_masks(
    model_path: str, image_path: str
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """
    Predict bboxes and masks from a YOLO model and return scores, bboxes and masks.
    Compatuble with Pydantic

    Args:
        model_path: Path to the YOLO model. Use segmentation model for box and mask prediction. Use pose model for pose prediction.
        Segmentation model name format: yolo11<size>-seg.pt
        image_path: Path to the image to predict

    Returns:
        scores: List of scores
        bboxes_xyxyn: List of bboxes
        masks: List of masks
    """
    model = YOLO(model_path)
    _, scores, bboxes_xyxyn, masks = predict_bboxes_and_masks_from_model(
        model, image_path
    )
    return scores, bboxes_xyxyn, masks


def predict_bboxes_and_masks_from_model(
    model: ultralytics.models.yolo.model.YOLO, image_path: str
) -> tuple[
    ultralytics.engine.results.Results,
    list[float],
    list[list[float]],
    list[list[float]],
]:
    """
    Predict bboxes and masks from a YOLO model and return YOLO results, scores, bboxes and masks.

    Args:
        model: YOLO Segmentation model
        image_path: Path to the image to predict

    Returns:
        result_ultra: YOLO results
        result_dict: Dictionary with bboxes, scores and masks
    """
    results_ultra = model([image_path])
    result_dict = {"bboxes": [], "scores": [], "masks": [], "classes": []}
    assert len(results_ultra) == 1, "Only one image is supported"
    result_ultra = results_ultra[0]
    bboxes = result_ultra.boxes
    result_dict["bboxes"] = [bbox.xyxyn.cpu().numpy().tolist() for bbox in bboxes]
    result_dict["scores"] = [score.item() for score in bboxes.conf]
    result_dict["masks"] = result_ultra.masks.data.cpu().numpy().tolist()
    result_dict["classes"] = [result_ultra.names[int(cls.item())] for cls in bboxes.cls]
    return result_ultra, result_dict


def predict_poses_from_model(
    model: ultralytics.models.yolo.model.YOLO, image_path: str
) -> tuple[
    ultralytics.engine.results.Results,
    list[float],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    """
    Predict poses from a YOLO model and return YOLO results, scores, bboxes and keypoints.
    """
    results_ultra = model([image_path])
    result_dict = {
        "bboxes": [],
        "scores": [],
        "masks": [],
        "classes": [],
        "keypoints": [],
        "keypoints_scores": [],
    }
    assert len(results_ultra) == 1, "Only one image is supported"
    result_ultra = results_ultra[0]
    bboxes = result_ultra.boxes
    keypoints = result_ultra.keypoints
    result_dict["bboxes"] = [bbox.xyxyn.cpu().numpy().tolist() for bbox in bboxes]
    result_dict["scores"] = [score.item() for score in bboxes.conf]
    result_dict["classes"] = [result_ultra.names[int(cls.item())] for cls in bboxes.cls]
    result_dict["keypoints"] = keypoints.data.cpu().numpy().tolist()
    result_dict["keypoints_scores"] = keypoints.conf.cpu().numpy().tolist()
    return result_ultra, result_dict


def main():
    args = do_parsing()
    model = YOLO(args.model_path)

    if args.model_path.endswith("-seg.pt"):
        result_ultra, result_dict = predict_bboxes_and_masks_from_model(
            model=model, image_path=args.image_path
        )
    else:
        result_ultra, result_dict = predict_poses_from_model(
            model=model, image_path=args.image_path
        )
    
    # TODO: Process the dictionary and build a custom visualizer function
        
    # Process results list
    if args.show_pred:
        result_ultra.show()
    if args.output_path:
        # Save to disk
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        result_ultra.save(filename=args.output_path)


if __name__ == "__main__":
    main()
