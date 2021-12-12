# Adapted from https://github.com/Cartucho/mAP

from collections import defaultdict
from typing import Any, Dict, List

from obj_det_metrics.utils import (
    _compute_counts_cumsum,
    _generate_dt_objs,
    _generate_gt_objs,
    _get_best_gt_bbox,
)
from obj_det_metrics.variables import (
    ClassName,
    DetectionsDict,
    GroundTruthDict,
    OutputsDict,
)


def _voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def compute_ap_map(
    ground_truth_dict_list: List[GroundTruthDict],
    detections_dict_list: List[DetectionsDict],
    iou_threshold: float = 0.5,
) -> OutputsDict:
    gt_file_ids = set([gt_dict["file_id"] for gt_dict in ground_truth_dict_list])

    gt_count_per_class, gt_bboxes_dict = _generate_gt_objs(ground_truth_dict_list, detections_dict_list)
    gt_classes = sorted(gt_count_per_class.keys())
    n_classes = len(gt_classes)

    dt_bboxes_dict = _generate_dt_objs(gt_classes, detections_dict_list, gt_file_ids)

    sum_ap = 0.0
    outputs_dict: Dict[str, Any] = {"ap": {}}
    true_positive_counts: Dict[ClassName, int] = defaultdict(lambda: 0)

    for class_name in gt_classes:
        dt_bboxes = dt_bboxes_dict[class_name]
        num_detections = len(dt_bboxes)
        # create arrays of zeros of size `num_detections`
        tp = [0] * num_detections
        fp = [0] * num_detections
        for idx, dt_bbox in enumerate(dt_bboxes):
            # Get corresponding ground truth bounding boxes
            gt_match, max_iou = _get_best_gt_bbox(dt_bbox, class_name, gt_bboxes_dict)
            if max_iou >= iou_threshold and not gt_match.matched:
                tp[idx] = 1
                gt_match.set_matched(True)
                true_positive_counts[class_name] += 1
            else:
                # false positive (multiple detections or low iou score)
                fp[idx] = 1

        _compute_counts_cumsum(fp)
        _compute_counts_cumsum(tp)
        rec = [val / gt_count_per_class[class_name] for val in tp]
        prec = [tp_val / (tp_val + fp_val) for tp_val, fp_val in zip(tp, fp)]

        ap, _, _ = _voc_ap(rec[:], prec[:])
        sum_ap += ap
        outputs_dict["ap"][class_name] = ap
    map_score = sum_ap / n_classes
    outputs_dict["map"] = map_score
    return outputs_dict
