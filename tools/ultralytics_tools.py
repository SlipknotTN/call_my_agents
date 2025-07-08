from ultralytics import YOLO
from tools.ultralytics_models import predict_bboxes_and_masks_from_model, predict_poses_from_model


def predict_bboxes_and_masks(
    model_path: str, image_path: str
) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """
    Predict bboxes and masks from a YOLO model and return scores, bboxes and masks.
    Compatible with Pydantic

    Args:
        model_path: Path to the YOLO model. Use segmentation model for box and mask prediction. Use pose model for pose prediction. Segmentation model name format: yolo11<size>-seg.pt, e.g. yolo11s-seg.pt
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


def predict_poses(model_path: str, image_path: str) -> dict[str, any]:
    """
    Predict human poses (persons) from a YOLO model and return scores, bboxes and keypoints.
    Compatible with Pydantic

    Args:
        model_path: Path to the YOLO model. Use pose model for pose prediction. Pose model name format: yolo11<size>-pose.pt, e.g. yolo11s-pose.pt
        image_path: Path to the image to predict

    Returns:
        Dictionary with bboxes, scores, classes, keypoints and keypoints scores
    """
    model = YOLO(model_path)
    result_ultra, result_dict = predict_poses_from_model(
        model=model, image_path=image_path
    )
    return result_dict
