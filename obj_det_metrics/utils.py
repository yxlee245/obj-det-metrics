from typing import Any, Dict, List, Tuple

from obj_det_metrics.variables import BoundingBox, ClassName, Coordinates


def read_file_lines_to_list(filepath: str) -> List[str]:
    """Helper funtion to load txt file lines to a list

    Args:
        filepath (str): Path of text file to read lines from

    Returns:
        List[str]: List of lines read from input text file
    """
    with open(filepath) as f:
        # remove whitespace characters like `\n` at the end of each line
        content = [line.strip() for line in f.readlines()]
    return content


def _generate_detections_by_class(result_dict: Dict[str, Any], class_name: ClassName) -> List[BoundingBox]:
    detections: List[BoundingBox] = []
    for idx in range(len(result_dict["labels"])):
        if result_dict["labels"][idx] == class_name:
            detections.append(
                BoundingBox(
                    conf_score=result_dict["scores"][idx],
                    class_name=result_dict["labels"][idx],
                    coordinates=result_dict["boxes"][idx],
                )
            )
    return detections


def _generate_ground_truths(ground_truth_dict: Dict[str, Any]) -> List[BoundingBox]:
    ground_truth_list: List[BoundingBox] = []
    for idx in range(len(ground_truth_dict["labels"])):
        ground_truth_list.append(
            BoundingBox(
                conf_score=1,
                class_name=ground_truth_dict["labels"][idx],
                coordinates=ground_truth_dict["boxes"][idx],
            )
        )
    return ground_truth_list


def _compute_iou(bbgt: Coordinates, bbdt: Coordinates) -> float:
    intersection_coordinates = [
        max(bbdt[0], bbgt[0]),
        max(bbdt[1], bbgt[1]),
        min(bbdt[2], bbgt[2]),
        min(bbdt[3], bbgt[3]),
    ]
    intersection_width = max(intersection_coordinates[2] - intersection_coordinates[0] + 1, 0)
    intersection_height = max(intersection_coordinates[3] - intersection_coordinates[1] + 1, 0)
    # compute overlap (IoU) = area of intersection / area of union
    union_area = (
        (bbdt[2] - bbdt[0] + 1) * (bbdt[3] - bbdt[1] + 1)
        + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
        - intersection_width * intersection_height
    )
    return intersection_width * intersection_height / union_area


def _compute_tp_fp_for_detections(
    count_true_positives: Dict[ClassName, int],
    class_name: ClassName,
    detections: List[BoundingBox],
    ground_truth_list: List[BoundingBox],
    iou_threshold: float,
) -> Tuple[List[float], List[float]]:
    count_true_positives[class_name] = 0
    tp = [0.0] * len(detections)
    fp = [0.0] * len(detections)

    for i, detection in enumerate(detections):
        max_iou = -1.0
        gt_match = BoundingBox(class_name="empty", coordinates=[0, 0, 0, 0])

        bbdt = detection.coordinates
        for j, detection in enumerate(ground_truth_list):
            if ground_truth_list[j].class_name == class_name:
                bbgt = ground_truth_list[j].coordinates
                iou = _compute_iou(bbgt, bbdt)
                if iou > max_iou:
                    max_iou = iou
                    gt_match = detection

        if max_iou >= iou_threshold and not gt_match.matched:

            # true positive
            tp[i] = 1
            gt_match.matched = True
            count_true_positives[class_name] += 1
            continue

        # false positive (multiple detection or iou less than threshold)
        fp[i] = 1

    return tp, fp


def _compute_cumsum(values: List[float]) -> None:
    cumsum = 0.0
    for idx, val in enumerate(values):
        values[idx] += cumsum
        cumsum += val
