"""
Code with YOLO related functions. The code is compatible with Pydantic and duplicated from ultralytics_models.py to simplify the LLM understanding of the code,
specifically the output format of the functions.

import tools.ultralytics_tools to use the functions in this file
"""
from ultralytics import YOLO


def predict_bboxes_and_masks(
    model_path: str, image_path: str
) -> dict[str, any]:
    """
    Predict bboxes and masks from a YOLO model and return scores, bboxes and masks.

    Args:
        model_path: Path to the YOLO model. Use segmentation model for box and mask prediction. Use pose model for pose prediction. Segmentation model name format: yolo11<size>-seg.pt, e.g. yolo11s-seg.pt
        image_path: Path to the image to predict

    Returns:
        Dictionary with bboxes, scores, masks and classes
    """
    model = YOLO(model_path)
    results_ultra = model([image_path])
    result_dict = {"bboxes": [], "scores": [], "masks": [], "classes": []}
    assert len(results_ultra) == 1, "Only one image is supported"
    result_ultra = results_ultra[0]
    bboxes = result_ultra.boxes
    result_dict["bboxes"] = bboxes.xyxyn.cpu().numpy().tolist()
    result_dict["scores"] = [score.item() for score in bboxes.conf]
    result_dict["masks"] = result_ultra.masks.data.cpu().numpy().tolist()
    result_dict["classes"] = [result_ultra.names[int(cls.item())] for cls in bboxes.cls]
    return result_dict


def predict_poses(model_path: str, image_path: str) -> dict[str, any]:
    """
    Predict human poses (persons) from a YOLO model and return scores, bboxes and keypoints.
    bboxes, scores, masks and classes are at the person level. The classes are always "person".
    keypoints and keypoints_scores are at the single keypoint level.

    Args:
        model_path: Path to the YOLO model. Use pose model for pose prediction. Pose model name format: yolo11<size>-pose.pt, e.g. yolo11s-pose.pt
        image_path: Path to the image to predict

    Returns:
        Dictionary with bboxes, scores, classes, keypoints and keypoints scores
    """
    model = YOLO(model_path)
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
    result_dict["bboxes"] = bboxes.xyxyn.cpu().numpy().tolist()
    result_dict["scores"] = [score.item() for score in bboxes.conf]
    result_dict["classes"] = [result_ultra.names[int(cls.item())] for cls in bboxes.cls]
    result_dict["keypoints"] = keypoints.data.cpu().numpy().tolist()
    result_dict["keypoints_scores"] = keypoints.conf.cpu().numpy().tolist()
    return result_dict
