import os

from obj_det_metrics.utils import (
    _generate_empty_detections_dict,
    _generate_empty_ground_truths_dict,
    _read_file_lines_to_list,
)
from obj_det_metrics.variables import Detections, GroundTruths


def read_detections_from_all_txts(txt_dir: str) -> Detections:
    detections_dict = _generate_empty_detections_dict()
    txt_filepaths = [os.path.join(txt_dir, filename) for filename in os.listdir(txt_dir)]
    for filepath in txt_filepaths:
        _read_detections_from_single_txt(filepath, detections_dict)
    return detections_dict


def _read_detections_from_single_txt(filepath: str, detections_dict: Detections):
    lines = _read_file_lines_to_list(filepath)
    for line in lines:
        line_contents = line.split()
        class_name = line_contents[0]
        conf_score = float(line_contents[1])
        coordinates = [int(content) for content in line_contents[2:]]
        detections_dict["boxes"].append(coordinates)
        detections_dict["labels"].append(class_name)
        detections_dict["scores"].append(conf_score)


def read_ground_truths_from_all_txts(txt_dir: str) -> GroundTruths:
    ground_truth_dict = _generate_empty_ground_truths_dict()
    txt_filepaths = [os.path.join(txt_dir, filename) for filename in os.listdir(txt_dir)]
    for filepath in txt_filepaths:
        _read_ground_truths_from_single_txt(filepath, ground_truth_dict)
    return ground_truth_dict


def _read_ground_truths_from_single_txt(filepath: str, ground_truth_dict: GroundTruths):
    lines = _read_file_lines_to_list(filepath)
    for line in lines:
        line_contents = line.split()
        class_name = line_contents[0]
        coordinates = [int(content) for content in line_contents[1:]]
        ground_truth_dict["boxes"].append(coordinates)
        ground_truth_dict["labels"].append(class_name)
