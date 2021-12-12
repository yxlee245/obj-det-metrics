from typing import List

import pytest

from obj_det_metrics.utils import (
    _compute_counts_cumsum,
    _compute_iou,
    _generate_dt_objs,
    _generate_gt_objs,
)
from obj_det_metrics.variables import BoundingBox, ClassName

GROUND_TRUTH_DICT_LIST = [
    {
        "coordinates": [[60, 80, 66, 92], [59, 94, 68, 97], [70, 87, 81, 94], [8, 34, 10, 36]],
        "class_labels": ["class2", "class3", "class3", "class4"],
        "file_id": "test1",
    },
    {
        "coordinates": [[111, 82, 164, 94], [54, 114, 68, 120], [110, 88, 112, 98], [92, 30, 100, 38]],
        "class_labels": ["class1", "class1", "class4", "class4"],
        "file_id": "test2",
    },
]

DETECTIONS_DICT_LIST = [
    {
        "coordinates": [[59, 82, 66, 94], [58, 94, 68, 95], [70, 88, 81, 93], [10, 34, 12, 38]],
        "class_labels": ["class2", "class3", "class3", "class4"],
        "conf_scores": [0.99056727, 0.98965424, 0.93990153, 0.9157755],
        "file_id": "test1",
    },
    {
        "coordinates": [[109, 82, 166, 94], [58, 114, 68, 115], [105, 88, 111, 98], [90, 30, 102, 38]],
        "class_labels": ["class1", "class1", "class4", "class4"],
        "conf_scores": [0.9823532, 0.8215353, 0.5923226, 0.9157755],
        "file_id": "test2",
    },
]

GT_CLASSES: List[ClassName] = ["class1", "class2", "class3", "class4"]
GT_FILE_IDS = {"test1", "test2"}


def test_generate_gt_objs():
    gt_count_per_class, gt_bboxes_dict = _generate_gt_objs(GROUND_TRUTH_DICT_LIST, DETECTIONS_DICT_LIST)
    assert set(gt_count_per_class.keys()) == set(
        GT_CLASSES
    ), f"Unexpected class name(s) in gt_count_per_class: {set(gt_count_per_class) - set(GT_CLASSES)}"
    assert gt_count_per_class["class1"] == 2, f"Expected 2 counts for class1 but got {gt_count_per_class['class1']}"
    assert gt_count_per_class["class2"] == 1, f"Expected 1 count for class2 but got {gt_count_per_class['class2']}"
    assert gt_count_per_class["class3"] == 2, f"Expected 2 counts for class3 but got {gt_count_per_class['class3']}"
    assert gt_count_per_class["class4"] == 3, f"Expected 3 counts for class4 but got {gt_count_per_class['class4']}"
    assert all(file_id in GT_FILE_IDS for file_id in gt_bboxes_dict), "Unexpected file ID(s) found in gt_bboxes_dict"
    assert all(
        isinstance(bbox, BoundingBox) for bboxes in gt_bboxes_dict.values() for bbox in bboxes
    ), "Wrong value type found in gt_bboxes_dict"


def test_generate_dt_objs():
    dt_bboxes_dict = _generate_dt_objs(GT_CLASSES, DETECTIONS_DICT_LIST, GT_FILE_IDS)
    assert all(
        class_name in GT_CLASSES for class_name in dt_bboxes_dict
    ), "Unexpected class name(s) found in dt_bboxes_dict"
    assert all(
        isinstance(bbox, BoundingBox) for bboxes in dt_bboxes_dict.values() for bbox in bboxes
    ), "Wrong value type found in dt_bboxes_dict"
    for bboxes in dt_bboxes_dict.values():
        conf_scores = [bbox.conf_score for bbox in bboxes]
        assert conf_scores == sorted(conf_scores, reverse=True), "Bounding boxes in dt_bboxes not sorted by conf_score"


@pytest.mark.parametrize(
    "dt_coordinates, gt_coordinates, expected_output",
    [
        ([0, 10, 20, 30], [0, 10, 20, 30], 1.0),
        ([0, 10, 20, 30], [30, 10, 50, 30], 0.0),
        ([0, 10, 20, 30], [0, 40, 20, 60], 0.0),
        ([0, 10, 20, 30], [0, 5, 10, 40], 0.381),
    ],
)
def test_compute_iou(dt_coordinates, gt_coordinates, expected_output):
    output = _compute_iou(dt_coordinates, gt_coordinates)
    assert (
        expected_output - 0.001 < output < expected_output + 0.001
    ), f"Expected IoU ~{expected_output} but got {output}"


@pytest.mark.parametrize("values, expected_output", [([0, 1, 2, 3], [0, 1, 3, 6]), ([1, 3, 5, 6], [1, 4, 9, 15])])
def test_compute_counts_cumsum(values, expected_output):
    _compute_counts_cumsum(values)
    assert values == expected_output
