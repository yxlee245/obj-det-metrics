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
        """Initialize class variables

        Args:
            coordinates (Coordinates): Coordinates of bounding box, in the form [xmin, ymin, xmax, ymax]
            class_name (ClassName): Class name of bounding box
            file_id (str): short filename (without file extension) of image where bounding box is found
            matched (bool, optional): Applies to ground truth box only. Flag to indicate if ground truth box has been
                matched to a detection box. Defaults to False.
            conf_score (float, optional): Applies to  detection bounding box only. Confidence score of bounding box.
                Defaults to 1.0.
        """
        self._coordinates = coordinates
        self._class_name = class_name
        self._file_id = file_id
        self._matched = matched
        self._conf_score = conf_score

    @property
    def coordinates(self) -> Coordinates:
        """Returns read-only bounding box coordinates in the form [xmin, ymin, xmax, ymax]

        Returns:
            Coordinates: Bounding box in the form [xmin, ymin, xmax, ymax]
        """
        return self._coordinates

    @property
    def class_name(self) -> ClassName:
        """Returns read-only class name of bounding box

        Returns:
            ClassName: Class name of bounding box
        """
        return self._class_name

    @property
    def file_id(self) -> str:
        """Returns read-only short filename of image that the bounding box belongs to

        Returns:
            str: Short filename of image that the bounding box belongs to
        """
        return self._file_id

    @property
    def matched(self) -> bool:
        """Returns read-only flag indicating if ground truth bounding box is matched to detection bounding box

        Returns:
            bool: Flag indicating if ground truth bounding box is matched to detection bounding box
        """
        return self._matched

    @property
    def conf_score(self) -> float:
        """Returns read-only confidence score for detection bounding box. Confidence score is always 1.0 for
            for ground truth bounding box

        Returns:
            float: Confidence score for detection bounding box
        """
        return self._conf_score

    def set_matched(self, matched: bool):
        """Method to set `matched` flag for ground truth bounding box

        Args:
            matched (bool): Flag value to be set for ground truth bounding box
        """
        self._matched = matched
