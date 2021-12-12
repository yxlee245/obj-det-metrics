# Adpated from https://github.com/Cartucho/mAP

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from obj_det_metrics.variables import (
    BoundingBox,
    ClassName,
    Coordinates,
    DetectionsDict,
    GroundTruthDict,
)


def _generate_gt_objs(
    ground_truth_dict_list: List[GroundTruthDict], detections_dict_list: List[DetectionsDict]
) -> Tuple[Dict[ClassName, int], Dict[str, List[BoundingBox]]]:
    """Helper function to generate:
    - `gt_count_per_class`: Dict containing total counts of ground truth bounding boxes per class
    - `gt_bboxes_dict`: Dict containing list of ground truth bounding box objects for each file ID

    Args:
        ground_truth_dict_list (List[GroundTruthDict]): List of dicts containing ground truth coordinates,
            class labels and file IDs
        detections_dict_list (List[DetectionsDict]): List of dicts containing detection coordinates,
            class labels, confidence scores and file IDs

    Returns:
        Tuple[Dict[ClassName, int], Dict[str, List[BoundingBox]]]: Contains `gt_count_per_class` and `gt_bboxes_dict`
    """
    gt_count_per_class: Dict[ClassName, int] = defaultdict(lambda: 0)
    dt_file_ids = set([dt_dict["file_id"] for dt_dict in detections_dict_list])
    gt_bboxes_dict: Dict[str, List[BoundingBox]] = defaultdict(lambda: [])
    for gt_dict in ground_truth_dict_list:
        # check if there is a corresponding detection-results file id
        assert gt_dict["file_id"] in dt_file_ids, f"File ID {gt_dict['file_id']} not found in detections list"
        # create bounding box object for ground truth
        gt_len = len(gt_dict["class_labels"])
        for i in range(gt_len):
            gt_bboxes_dict[gt_dict["file_id"]].append(
                BoundingBox(
                    coordinates=gt_dict["coordinates"][i],
                    class_name=gt_dict["class_labels"][i],
                    file_id=gt_dict["file_id"],
                    conf_score=1.0,
                )
            )
            gt_count_per_class[gt_dict["class_labels"][i]] += 1
    return gt_count_per_class, gt_bboxes_dict


def _generate_dt_objs(
    gt_classes: List[ClassName], detections_dict_list: List[DetectionsDict], gt_file_ids: Set[str]
) -> Dict[ClassName, List[BoundingBox]]:
    """Helper function to generate:
    - `dt_bboxes_dict`: : Dict containing list of detection bounding box objects for each class

    Args:
        gt_classes (List[ClassName]): List of unique class labels in ground truth
        detections_dict_list (List[DetectionsDict]): List of dicts containing detection coordinates,
            class labels, confidence scores and file IDs
        gt_file_ids (Set[str]): Set of unique file IDs for ground truth

    Returns:
        Dict[ClassName, List[BoundingBox]]: Dict `dt_bboxes_dict` containing list of detection bounding box objects
            for each class
    """
    dt_bboxes_dict: Dict[ClassName, List[BoundingBox]] = defaultdict(lambda: [])
    for class_name in gt_classes:
        for dt_dict in detections_dict_list:
            # check if there is a corresponding ground truth file id
            assert dt_dict["file_id"] in gt_file_ids, f"File ID {dt_dict['file_id']} not found in ground truth list"
            dt_len = len(dt_dict["class_labels"])
            dt_bboxes_dict[class_name].extend(
                [
                    BoundingBox(
                        coordinates=dt_dict["coordinates"][i],
                        class_name=dt_dict["class_labels"][i],
                        file_id=dt_dict["file_id"],
                        conf_score=dt_dict["conf_score"][i],
                    )
                    for i in range(dt_len)
                    if dt_dict["class_labels"][i] == class_name
                ]
            )
        dt_bboxes_dict[class_name].sort(key=lambda bbox: bbox.conf_score, reverse=True)

    return dt_bboxes_dict


def _compute_iou(dt_coordinates: Coordinates, gt_coordinates: Coordinates) -> float:
    """Helper function to compute IoU

    Args:
        dt_coordinates (Coordinates): Coordinates of detection bounding box, in the form [xmin, ymin, xmax, ymax]
        gt_coordinates (Coordinates): Coordinates of ground truth box, in the form [xmin, ymin, xmax, ymax]

    Returns:
        float: IoU score
    """
    int_coordintates = [
        max(dt_coordinates[0], gt_coordinates[0]),
        max(dt_coordinates[1], gt_coordinates[1]),
        min(dt_coordinates[2], gt_coordinates[2]),
        min(dt_coordinates[3], gt_coordinates[3]),
    ]
    int_width = max(int_coordintates[2] - int_coordintates[0] + 1, 0)
    int_height = max(int_coordintates[3] - int_coordintates[1] + 1, 0)
    int_area = int_width * int_height
    dt_area = (dt_coordinates[2] - dt_coordinates[0] + 1) * (dt_coordinates[3] - dt_coordinates[1] + 1)
    gt_area = (gt_coordinates[2] - gt_coordinates[0] + 1) * (gt_coordinates[3] - gt_coordinates[1] + 1)
    union_area = dt_area + gt_area - int_area
    return int_area / union_area


def _get_best_gt_bbox(
    dt_bbox: BoundingBox, class_name: ClassName, gt_bboxes_dict: Dict[str, List[BoundingBox]]
) -> Tuple[BoundingBox, float]:
    """Helper function to select best ground truth bounding box and compute best IoU score for
    inputted detecton bounding box

    Args:
        dt_bbox (BoundingBox): Detection bounding box object
        class_name (ClassName): Class name to be used for comparison
        gt_bboxes_dict (Dict[str, List[BoundingBox]]): Dict containing list of ground truth bounding box objects for
            each file ID

    Returns:
        Tuple[BoundingBox, float]: Contains best ground truth bounding box object and best IoU score
    """
    dt_file_id = dt_bbox.file_id
    gt_bboxes = gt_bboxes_dict[dt_file_id]
    max_iou = -1.0
    gt_match = BoundingBox([0, 0, 0, 0], class_name="empty", file_id=dt_file_id, conf_score=0.0)
    dt_coordinates = dt_bbox.coordinates
    for gt_bbox in gt_bboxes:
        if gt_bbox.class_name == class_name:
            gt_coordinates = gt_bbox.coordinates
            iou = _compute_iou(dt_coordinates, gt_coordinates)
            if iou > max_iou:
                max_iou = iou
                gt_match = gt_bbox
    return gt_match, max_iou


def _compute_counts_cumsum(values: List[int]):
    """Helper function to compute cumulative sum of count values in-memory

    Args:
        values (List[int]): List of counts to compute cumulative sum over
    """
    cumsum = 0
    for idx, val in enumerate(values):
        values[idx] += cumsum
        cumsum += val
