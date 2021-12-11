from obj_det_metrics.ap_map import calculate_ap_map, calculate_ap_map_range

ground_truth = {
    "boxes": [
        [60.0, 80.0, 66.0, 92.0],
        [59.0, 94.0, 68.0, 97.0],
        [70.0, 87.0, 81.0, 94.0],
        [8.0, 34.0, 10.0, 36.0],
    ],
    "labels": ["class2", "class3", "class3", "class4"],
}

result_dict = {
    "boxes": [
        [59.0, 82.0, 66.0, 94.0],
        [58.0, 94.0, 68.0, 95.0],
        [70.0, 88.0, 81.0, 93.0],
        [10.0, 34.0, 12.0, 38.0],
    ],
    "labels": ["class2", "class3", "class3", "class4"],
    "scores": [0.99056727, 0.98965424, 0.93990153, 0.9157755],
}

# calculates the mAP for an IOU threshold of 0.5
print(calculate_ap_map(ground_truth, result_dict, 0.5))

# calculates the mAP average for the IOU thresholds 0.05, 0.1, 0.15, ..., 0.90, 0.95.
print(calculate_ap_map_range(ground_truth, result_dict, 0.05, 0.95, 0.05))
