from obj_det_metrics.ap_map import compute_ap_map


def main():
    ground_truth_dict_list = [
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
    detections_dict_list = [
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

    outputs_dict = compute_ap_map(ground_truth_dict_list, detections_dict_list)

    print("# APs #")
    for class_name, ap in outputs_dict["ap"].items():
        print(f"{class_name}: {ap * 100:0.2f}%")
    print("# mAP #")
    print(f"{outputs_dict['map'] * 100:0.2f}%")


if __name__ == "__main__":
    main()
