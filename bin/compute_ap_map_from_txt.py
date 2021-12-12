from obj_det_metrics.ap_map import compute_ap_map
from obj_det_metrics.ingest import (
    generate_dt_dict_list_from_txts,
    generate_gt_dict_list_from_txts,
)


def main():
    ground_truth_dict_list = generate_gt_dict_list_from_txts("tests/fixtures/test_ground_truths")
    detections_dict_list = generate_dt_dict_list_from_txts("tests/fixtures/test_detections")

    outputs_dict = compute_ap_map(ground_truth_dict_list, detections_dict_list)

    print("# APs #")
    for class_name, ap in outputs_dict["ap"].items():
        print(f"{class_name}: {ap * 100:0.2f}%")
    print("# mAP #")
    print(f"{outputs_dict['map'] * 100:0.2f}%")


if __name__ == "__main__":
    main()
