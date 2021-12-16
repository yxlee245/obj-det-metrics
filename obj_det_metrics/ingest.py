import os
from typing import List

import pipe

from obj_det_metrics.utils import _generate_empty_dt_dict, _generate_empty_gt_dict
from obj_det_metrics.variables import DetectionsDict, GroundTruthDict


def _read_file_lines(filepath: str) -> List[str]:
    """Helper function to read in lines from a text file, and strip whitespaces before and after each line

    Args:
        filepath (str): Path of text file to read lines from

    Returns:
        List[str]: Lines contained in text file
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def _generate_gt_dict_from_txt(filepath: str) -> GroundTruthDict:
    """Helper function to generate a dict containing ground truth from a text file, where each line is in the format
    "<class name> <xmin> <ymin> <xmax> <ymax>" (adapted from https://github.com/Cartucho/mAP).

    Args:
        filepath (str): Path of text file to read lines from

    Returns:
        GroundTruthDict: Dict containing coordinates, class labels and file ID of single image
    """
    lines = _read_file_lines(filepath)
    gt_dict = _generate_empty_gt_dict()
    gt_dict["file_id"] = os.path.splitext(os.path.basename(filepath))[0]
    for line in lines:
        line_contents = line.split()
        class_label = line_contents[0]
        coordinates = [int(value) for value in line_contents[1:]]
        gt_dict["class_labels"].append(class_label)
        gt_dict["coordinates"].append(coordinates)
    return gt_dict


def generate_gt_dict_list_from_txts(txt_dir: str) -> List[GroundTruthDict]:
    """Overall function to read in text files from the inputted directory and return a list of ground truth dicts

    Args:
        txt_dir (str): Directory containing text files to be read

    Returns:
        List[GroundTruthDict]: List of ground truth dicts
    """
    gt_dict_list = list(
        os.listdir(txt_dir)
        | pipe.where(lambda filename: ".txt" in filename.lower())
        | pipe.select(lambda filename: os.path.join(txt_dir, filename))
        | pipe.select(_generate_gt_dict_from_txt)
    )
    return gt_dict_list


def _generate_dt_dict_from_txt(filepath: str) -> DetectionsDict:
    """Helper function to generate a dict containing detections from a text file, where each line is in the format
    "<class name> <conf_score> <xmin> <ymin> <xmax> <ymax>" (adapted from https://github.com/Cartucho/mAP).

    Args:
        filepath (str): Path of text file to read lines from

    Returns:
        DetectionsDict: Dict containing coordinates, class labels and file ID of single image
    """
    lines = _read_file_lines(filepath)
    dt_dict = _generate_empty_dt_dict()
    dt_dict["file_id"] = os.path.splitext(os.path.basename(filepath))[0]
    for line in lines:
        line_contents = line.split()
        class_label = line_contents[0]
        conf_score = float(line_contents[1])
        coordinates = [int(value) for value in line_contents[2:]]
        dt_dict["class_labels"].append(class_label)
        dt_dict["conf_scores"].append(conf_score)
        dt_dict["coordinates"].append(coordinates)
    return dt_dict


def generate_dt_dict_list_from_txts(txt_dir: str) -> List[DetectionsDict]:
    """Overall function to read in text files from the inputted directory and return a list of detections dicts

    Args:
        txt_dir (str): Directory containing text files to be read

    Returns:
        List[DetectionsDict]: List of ground truth dicts
    """
    dt_dict_list = list(
        os.listdir(txt_dir)
        | pipe.where(lambda filename: ".txt" in filename.lower())
        | pipe.select(lambda filename: os.path.join(txt_dir, filename))
        | pipe.select(_generate_dt_dict_from_txt)
    )
    return dt_dict_list
