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
        """Initialize class variables

        Args:
            class_name (ClassName): Class name of bounding box
            coordinates (Coordinates): Coordinates of bounding box, in the form [xmin, ymin, xmax, ymax]
            conf_score (float, optional): Confidence score of bounding box. Defaults to 1.0.
            matched (bool, optional): Applies to ground truth bounding boxes only. Indicates if ground truth box is
                matched with detection bounding box. Defaults to False.
        """
        self._coordinates = coordinates
        self._class_name = class_name
        self._conf_score = conf_score
        self._matched = matched

    def set_matched(self, matched: bool):
        """Setter method to set value of `self._matched`

        Args:
            matched (bool): [description]
        """
        self._matched = matched

    @property
    def coordinates(self) -> Coordinates:
        """Returns read-only version of bounding box coordinates

        Returns:
            Coordinates: Bounding box coordinates, in the form [xmin, ymin, xmax, ymax]
        """
        return self._coordinates

    @property
    def class_name(self) -> ClassName:
        """Returns read-only version of bounding box class name

        Returns:
            ClassName: Bounding box class name
        """
        return self._class_name

    @property
    def conf_score(self) -> float:
        """Returns read-only version of confidence score

        Returns:
            float: Confidence score
        """
        return self._conf_score

    @property
    def matched(self) -> bool:
        """Returns read-only version of matched flag

        Returns:
            bool: Flag that indicates if ground truth bounding box is matched with detection bounding box
        """
        return self._matched
