# Adapted from https://github.com/LeMuecke/mapcalc

from typing import Any, Dict, List, Union

ClassName = Union[str, int]
GroundTruthDict = Dict[str, Any]
DetectionsDict = Dict[str, Any]
Coordinates = List[Union[int, float]]
OutputsDict = Dict[str, Any]


class BoundingBox:
    def __init__(
        self,
        coordinates: Coordinates,
        class_name: ClassName,
        file_id: str,
        matched: bool = False,
        conf_score: float = 1.0,
    ) -> None:
        self._coordinates = coordinates
        self._class_name = class_name
        self._file_id = file_id
        self._matched = matched
        self._conf_score = conf_score

    @property
    def coordinates(self) -> Coordinates:
        return self._coordinates

    @property
    def class_name(self) -> ClassName:
        return self._class_name

    @property
    def file_id(self) -> str:
        return self._file_id

    @property
    def matched(self) -> bool:
        return self._matched

    @property
    def conf_score(self) -> float:
        return self._conf_score

    def set_matched(self, matched: bool):
        self._matched = matched
