from typing import List, Union

Coordinates = List[Union[int, float]]


class BoundingBox:
    def __init__(
        self,
        class_name: str,
        coordinates: Coordinates,
        conf_score: float = 1.0,
        matched: bool = False,
    ):
        self.coordinates = coordinates
        self.class_name = class_name
        self.conf_score = conf_score
        self.matched = matched
