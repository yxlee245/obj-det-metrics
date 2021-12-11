"""
Adapted from https://github.com/LeMuecke/mapcalc
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from obj_det_metrics.utils import (
    _compute_cumsum,
    _compute_tp_fp_for_detections,
    _generate_detections_by_class,
    _generate_ground_truths,
)
from obj_det_metrics.variables import ClassName


def _voc_ap(rec: List[float], prec: List[float]) -> Tuple[float, List[float], List[float]]:
    """Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
        precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.

    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

    Args:
        rec (List[float]): List of recall values
        prec (List[float]): List of precision values

    Returns:
        Tuple[float, List[float], List[float]]: Tuple containing average precision, mean recall and
            mean precision values
    """
    # insert 0.0 at beginning and 1.0 at end of list
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    # insert 0.0 at beginning and end of list
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    #  This part makes the precision monotonically decreasing
    #     (goes from the end to the beginning)
    #     matlab: for i=numel(mpre)-1:-1:1
    #                 mpre(i)=max(mpre(i),mpre(i+1));

    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    #  This part creates a list of indexes where the recall changes
    #     matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1

    #  The Average Precision (AP) is the area under the curve
    #     (numerical integration)
    #     matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def _check_dicts_for_content_and_size(ground_truth_dict: Dict[str, Any], result_dict: Dict[str, Any]):
    """Checks if the content and the size of the arrays adds up.
    Raises and exception if not, does nothing if everything is ok.

    Args:
        ground_truth_dict (Dict[str, Any]): Dict with {boxes:, labels:}
        e.g.
        {
        'boxes':
            [[60., 80., 66., 92.],
            [59., 94., 68., 97.],
            [70., 87., 81., 94.],
            [8., 34., 10., 36.]],

        'labels':
            [2, 2, 3, 4]}
        result_dict (Dict[str, Any]): Dict with {boxes:, labels:, scores:}
        e.g.
        {
        'boxes':
            [[57., 87., 66., 94.],
                [58., 94., 68., 95.],
                [70., 88., 81., 93.],
                [10., 37., 17., 40.]],

        'labels':
            [2, 3, 3, 4],

        'scores':
            [0.99056727, 0.98965424, 0.93990153, 0.9157755]}

    Raises:
        KeyError: Error raised if keys "boxes" or "labels" not found in `ground_truth_dict`
        KeyError: Error raised if keys "boxes" or "labels" not found in `result_dict`
        ValueError: Error raised if number of boxes and labels differ in `ground_truth_dict`
        ValueError: Error raised if number of boxes and labels differ in `result_dict`
    """
    if "boxes" not in ground_truth_dict or "labels" not in ground_truth_dict:
        raise KeyError("ground_truth_dict expects the keys 'boxes' and 'labels'.")
    if "boxes" not in result_dict or "labels" not in result_dict:
        raise KeyError("result_dict expects the keys 'boxes' and 'labels' and optionally 'scores'.")

    if "scores" not in result_dict:
        result_dict["scores"] = [1] * len(result_dict["boxes"])

    if len(ground_truth_dict["boxes"]) != len(ground_truth_dict["labels"]):
        raise ValueError("The number of boxes and labels differ in the ground_truth_dict.")

    if not len(result_dict["boxes"]) == len(result_dict["labels"]) == len(result_dict["scores"]):
        raise ValueError("The number of boxes, labels and scores differ in the result_dict.")


def calculate_ap_map(
    ground_truth_dict: Dict[str, Any], result_dict: Dict[str, Any], iou_threshold: float
) -> Dict[str, Any]:
    """Compute AP@[iou_threshold] for all classes and mAP@[iou_threshold]

    Args:
        ground_truth_dict (Dict[str, Any]): Dict with {boxes:, labels:}
        e.g.
        {
        'boxes':
            [[60., 80., 66., 92.],
                [59., 94., 68., 97.],
                [70., 87., 81., 94.],
                [8., 34., 10., 36.]],

        'labels':
            [2, 2, 3, 4]}
        result_dict (Dict[str, Any]): Dict with {boxes:, labels:, scores:}
        e.g.
        {
        'boxes':
            [[57., 87., 66., 94.],
            [58., 94., 68., 95.],
            [70., 88., 81., 93.],
            [10., 37., 17., 40.]],

        'labels':
            [2, 3, 3, 4],

        'scores':
            [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
        iou_threshold (float): minimum iou for which the detection counts as true positive

    Returns:
        Dict[str, Any]: Dict containing APs for all classes and mAP
    """
    # checking if the variables have the correct keys

    _check_dicts_for_content_and_size(ground_truth_dict, result_dict)

    occurring_gt_classes = set(ground_truth_dict["labels"])
    unique, counts = np.unique(ground_truth_dict["labels"], return_counts=True)
    ground_truth_counter_per_class = dict(zip(unique, counts))
    count_true_positives: Dict[ClassName, int] = {}
    sum_average_precision = 0.0

    ground_truth_list = _generate_ground_truths(ground_truth_dict)
    outputs_dict: Dict[str, Any] = {
        "ap": {class_name: 0.0 for class_name in occurring_gt_classes},
        "map": 0.0,
    }

    for class_name in occurring_gt_classes:

        detections_with_certain_class = _generate_detections_by_class(result_dict, class_name)

        count_true_positives[class_name] = 0

        tp, fp = _compute_tp_fp_for_detections(
            count_true_positives,
            class_name,
            detections_with_certain_class,
            ground_truth_list,
            iou_threshold,
        )

        # compute precision/recall
        _compute_cumsum(fp)
        _compute_cumsum(tp)

        rec = tp[:]
        for idx in range(len(tp)):
            rec[idx] = float(tp[idx]) / ground_truth_counter_per_class[class_name]

        prec = tp[:]
        for idx in range(len(tp)):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        average_precision, _, _ = _voc_ap(rec[:], prec[:])
        sum_average_precision += average_precision
        outputs_dict["ap"][class_name] = average_precision

    mean_average_precision = sum_average_precision / len(occurring_gt_classes)
    outputs_dict["map"] = mean_average_precision
    return outputs_dict


def calculate_ap_map_range(
    ground_truth_dict: Dict[str, Any],
    result_dict: Dict[str, Any],
    iou_begin: float,
    iou_end: float,
    iou_step: float,
) -> Dict[str, Any]:
    """Computes AP@[iou_begin:iou_end:iou_step] for all classes and mAP@[iou_begin:iou_end:iou_step],
        including iou_begin and iou_end.

    Args:
        ground_truth_dict (Dict[str, Any]): Dict with {boxes:, labels:}
        e.g.
        {
        'boxes':
            [[60., 80., 66., 92.],
            [59., 94., 68., 97.],
            [70., 87., 81., 94.],
            [8., 34., 10., 36.]],

        'labels':
            [2, 2, 3, 4]}
        result_dict (Dict[str, Any]): Dict with {boxes:, labels:, scores:}
        e.g.
        {
        'boxes':
            [[57., 87., 66., 94.],
            [58., 94., 68., 95.],
            [70., 88., 81., 93.],
            [10., 37., 17., 40.]],

        'labels':
            [2, 3, 3, 4],

        'scores':
            [0.99056727, 0.98965424, 0.93990153, 0.9157755]}
        iou_begin (float): first iou threshold to use for evaluation
        iou_end (float): last iou threshold to use for evaluation (included!)
        iou_step (float): step size

    Returns:
        Dict[str, Any]: Dict containing APs and mAP averaged over all IOU threshold values
    """

    _check_dicts_for_content_and_size(ground_truth_dict, result_dict)
    iou_list = np.arange(iou_begin, iou_end + iou_step, iou_step)
    occurring_gt_classes = set(ground_truth_dict["labels"])
    outputs_dict: Dict[str, Any] = {
        "mean_ap": {class_name: 0.0 for class_name in occurring_gt_classes},
        "mean_map": 0.0,
    }

    for iou in iou_list:
        ap_map_dict = calculate_ap_map(ground_truth_dict, result_dict, iou)
        for class_name, ap in ap_map_dict["ap"].items():
            outputs_dict["mean_ap"][class_name] += ap
        outputs_dict["mean_map"] += ap_map_dict["map"]

    for class_name in ap_map_dict["ap"].keys():
        outputs_dict["mean_ap"][class_name] /= len(iou_list)
    outputs_dict["mean_map"] /= len(iou_list)

    return outputs_dict
