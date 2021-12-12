from obj_det_metrics.ap_map import compute_ap_map

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


def test_compute_ap_map():
    output = compute_ap_map(GROUND_TRUTH_DICT_LIST, DETECTIONS_DICT_LIST, iou_threshold=0.5)
    assert 0.4791 < output["map"] < 0.4792, "Wrong value of mAP outputted, check implementation of mAP"
    assert len(output["ap"]) == 4, f"Expected 4 classes for APs, but got {len(output['ap'])} classes instead"
    assert 0.49 < output["ap"]["class1"] < 0.51, "Wrong AP for class1"
    assert 0.99 < output["ap"]["class2"] < 1.01, "Wrong AP for class2"
    assert 0.24 < output["ap"]["class3"] < 0.26, "Wrong AP for class3"
    assert 0.16 < output["ap"]["class4"] < 0.17, "Wrong AP for class4"
