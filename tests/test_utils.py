from typing import Any, Dict, List

import pytest

from obj_det_metrics.utils import (
    _compute_cumsum,
    _compute_iou,
    _generate_detections_by_class,
    _generate_ground_truths,
    _read_file_lines_to_list,
)

TEST_LINES_PATH = "tests/fixtures/test_lines.txt"

RESULT_DICT: Dict[str, List[Any]] = {
    "boxes": [
        [59.0, 82.0, 66.0, 94.0],
        [58.0, 94.0, 68.0, 95.0],
        [70.0, 88.0, 81.0, 93.0],
        [10.0, 34.0, 12.0, 38.0],
    ],
    "labels": ["class2", "class3", "class3", "class4"],
    "scores": [0.99056727, 0.98965424, 0.93990153, 0.9157755],
}

GROUND_TRUTH: Dict[str, List[Any]] = {
    "boxes": [
        [60.0, 80.0, 66.0, 92.0],
        [59.0, 94.0, 68.0, 97.0],
        [70.0, 87.0, 81.0, 94.0],
        [8.0, 34.0, 10.0, 36.0],
    ],
    "labels": ["class2", "class3", "class3", "class4"],
}


def test_read_file_lines_to_list():
    output = _read_file_lines_to_list(TEST_LINES_PATH)
    assert output == [
        "line 1",
        "line 2",
        "line 3",
    ], "Wrong implementation of read_file_lines_to_list()"


@pytest.mark.parametrize("class_name, expected_output_len", [("class2", 1), ("class3", 2), ("class4", 1)])
def test_generate_detections_by_class(class_name, expected_output_len):
    detections = _generate_detections_by_class(RESULT_DICT, class_name)
    assert (
        len(detections) == expected_output_len
    ), f"Wrong number of detections returned, expected {expected_output_len} but got {len(detections)}"
    assert all(
        0.0 <= detection.conf_score <= 1.0 for detection in detections
    ), "Invalid conf_score value for one or more detections"
    assert all(
        detection.class_name == class_name for detection in detections
    ), "Invalid class_name value for one or more detections"
    assert all(
        len(detection.coordinates) == 4 for detection in detections
    ), "Invalid number of coordinates for one or more detections"


def test_generate_ground_truths():
    ground_truth_list = _generate_ground_truths(GROUND_TRUTH)
    expected_len = len(GROUND_TRUTH["labels"])
    assert (
        len(ground_truth_list) == expected_len
    ), f"Wrong number of ground truths returned, expected {expected_len} but got {len(ground_truth_list)}"
    assert all(
        ground_truth.conf_score == 1.0 for ground_truth in ground_truth_list
    ), "Invalid conf_score value for one or more ground truths"
    assert all(
        len(ground_truth.coordinates) == 4 for ground_truth in ground_truth_list
    ), "Invalid number of coordinates for one or more ground truths"


@pytest.mark.parametrize(
    "bbgt, bbdt, expected_output",
    [
        ([0, 1, 2, 3], [4, 5, 6, 7], 0.0),
        ([0, 1, 3, 4], [2, 3, 5, 6], 1 / 7),
        ([0, 1, 2, 3], [0, 1, 2, 3], 1.0),
    ],
)
def test_compute_iou(bbgt, bbdt, expected_output):
    output = _compute_iou(bbgt, bbdt)
    assert output == expected_output, f"Expected IOU to be {expected_output} but got {output}"


@pytest.mark.parametrize(
    "values, expected_output",
    [([0.0, 0.1, 0.2], [0.0, 0.1, 0.3]), ([0.1, 0.3, 0.4], [0.1, 0.4, 0.8])],
)
def test_compute_cumsum(values, expected_output):
    _compute_cumsum(values)
    assert len(values) == len(expected_output), f"Expected {len(expected_output)} values but got {len(values)} values"
    assert all(
        abs(val - expected_val) < 1e-6 for val, expected_val in zip(values, expected_output)
    ), "Wrong computation of cumsum"
