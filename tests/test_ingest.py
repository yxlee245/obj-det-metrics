import pytest

from obj_det_metrics.ingest import (
    _generate_dt_dict_from_txt,
    _generate_gt_dict_from_txt,
    _read_file_lines,
    generate_dt_dict_list_from_txts,
    generate_gt_dict_list_from_txts,
)

GT_DIR = "tests/fixtures/test_ground_truths"
DT_DIR = "tests/fixtures/test_detections"


@pytest.mark.parametrize(
    "filepath, expected_output",
    [
        (
            "tests/fixtures/test_detections/test1.txt",
            [
                "class2 0.99056727 59 82 66 94",
                "class3 0.98965424 58 94 68 95",
                "class3 0.93990153 70 88 81 93",
                "class4 0.9157755 10 34 12 38",
            ],
        ),
        (
            "tests/fixtures/test_ground_truths/test1.txt",
            [
                "class2 60 80 66 92",
                "class3 59 94 68 97",
                "class3 70 87 81 94",
                "class4 8 34 10 36",
            ],
        ),
    ],
)
def test_read_file_lines(filepath, expected_output):
    output = _read_file_lines(filepath)
    assert output == expected_output, "Wrong output read from file"


@pytest.mark.parametrize(
    "filepath, expected_output",
    [
        (
            "tests/fixtures/test_ground_truths/test1.txt",
            {
                "coordinates": [[60, 80, 66, 92], [59, 94, 68, 97], [70, 87, 81, 94], [8, 34, 10, 36]],
                "class_labels": ["class2", "class3", "class3", "class4"],
                "file_id": "test1",
            },
        ),
        (
            "tests/fixtures/test_ground_truths/test2.txt",
            {
                "coordinates": [[111, 82, 164, 94], [54, 114, 68, 120], [110, 88, 112, 98], [92, 30, 100, 38]],
                "class_labels": ["class1", "class1", "class4", "class4"],
                "file_id": "test2",
            },
        ),
    ],
)
def test_generate_gt_dict_from_txt(filepath, expected_output):
    output = _generate_gt_dict_from_txt(filepath)
    assert output == expected_output, "Wrong output read from file for ground truth"


def test_generate_gt_dict_list_from_txts():
    output = generate_gt_dict_list_from_txts(GT_DIR)
    assert len(output) == 2, f"Expected 2 ground truth dicts but got {len(output)}"


@pytest.mark.parametrize(
    "filepath, expected_output",
    [
        (
            "tests/fixtures/test_detections/test1.txt",
            {
                "coordinates": [[59, 82, 66, 94], [58, 94, 68, 95], [70, 88, 81, 93], [10, 34, 12, 38]],
                "class_labels": ["class2", "class3", "class3", "class4"],
                "conf_scores": [0.99056727, 0.98965424, 0.93990153, 0.9157755],
                "file_id": "test1",
            },
        ),
        (
            "tests/fixtures/test_detections/test2.txt",
            {
                "coordinates": [[109, 82, 166, 94], [58, 114, 68, 115], [105, 88, 111, 98], [90, 30, 102, 38]],
                "class_labels": ["class1", "class1", "class4", "class4"],
                "conf_scores": [0.9823532, 0.8215353, 0.5923226, 0.9157755],
                "file_id": "test2",
            },
        ),
    ],
)
def test_generate_dt_dict_from_txt(filepath, expected_output):
    output = _generate_dt_dict_from_txt(filepath)
    assert output == expected_output, "Wrong output read from file for detections"


def test_generate_dt_dict_list_from_txts():
    output = generate_dt_dict_list_from_txts(DT_DIR)
    assert len(output) == 2, f"Expected 2 detections dicts but got {len(output)}"
