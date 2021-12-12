from typing import List, Union

ClassName = Union[str, int]
Coordinates = List[Union[int, float]]


class BoundingBox:
    def __init__(
        self,
        class_name: ClassName,
        coordinates: Coordinates,
        conf_score: float = 1.0,
        matched: bool = False,
    ):
        self._coordinates = coordinates
        self._class_name = class_name
        self._conf_score = conf_score
        self._matched = matched

    def set_matched(self, matched: bool):
        self._matched = matched

    @property
    def coordinates(self) -> Coordinates:
        return self._coordinates

    @property
    def class_name(self) -> ClassName:
        return self._class_name

    @property
    def conf_score(self) -> float:
        return self._conf_score

    @property
    def matched(self) -> bool:
        return self._matched
