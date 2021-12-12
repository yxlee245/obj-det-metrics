from typing import Any, Dict, List

from obj_det_metrics.utils import _read_file_lines_to_list


def read_detections_from_txt(filepath: str) -> Dict[str, List[Any]]:
    lines = _read_file_lines_to_list(filepath)
    output: Dict[str, List[Any]] = {"boxes": [], "labels": [], "scores": []}
    for line in lines:
        line_contents = line.split()
        class_name = line_contents[0]
        conf_score = float(line_contents[1])
        coordinates = [int(content) for content in line_contents[2:]]
        output["boxes"].append(coordinates)
        output["labels"].append(class_name)
        output["scores"].append(conf_score)
    return output
