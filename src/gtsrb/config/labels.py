"""Canonical GTSRB class definitions.

The 43 class labels live here and nowhere else. Both the evaluation code and the
inference API import them from this module, which is what keeps a prediction
returned by the API described with the same name the evaluator scores it under.

Class ids are the official GTSRB ids (``0``-``42``) and are stable: the dataset's
``Test.csv``/``Train.csv`` metadata and ``torchvision.datasets.GTSRB`` both use
them, and every checkpoint stores the ordering it was trained with so a mismatch
can be detected at load time rather than at prediction time.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Literal

NUM_CLASSES: Final[int] = 43

ClassCategory = Literal["speed_limit", "prohibition", "priority", "warning", "mandatory", "other"]

#: Official GTSRB class names, indexed by class id.
CLASS_NAMES: Final[tuple[str, ...]] = (
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
)

#: Compact labels for plot axes and API payloads where the full name is too long.
CLASS_SHORT_NAMES: Final[tuple[str, ...]] = (
    "Speed limit 20",
    "Speed limit 30",
    "Speed limit 50",
    "Speed limit 60",
    "Speed limit 70",
    "Speed limit 80",
    "End speed limit 80",
    "Speed limit 100",
    "Speed limit 120",
    "No passing",
    "No passing >3.5t",
    "Right-of-way",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles >3.5t prohibited",
    "No entry",
    "General caution",
    "Curve left",
    "Curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End all limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing >3.5t",
)

#: Sign family per class. Used by error analysis to separate "the model confused
#: two speed limits" (an inherently hard, fine-grained mistake) from "the model
#: confused a speed limit with a warning triangle" (a qualitatively different and
#: more serious failure). Reasoning is only as good as this grouping, so it is
#: explicit and reviewable rather than inferred from the class name.
CLASS_CATEGORIES: Final[tuple[ClassCategory, ...]] = (
    "speed_limit",  # 0  Speed limit 20
    "speed_limit",  # 1  Speed limit 30
    "speed_limit",  # 2  Speed limit 50
    "speed_limit",  # 3  Speed limit 60
    "speed_limit",  # 4  Speed limit 70
    "speed_limit",  # 5  Speed limit 80
    "speed_limit",  # 6  End of speed limit 80
    "speed_limit",  # 7  Speed limit 100
    "speed_limit",  # 8  Speed limit 120
    "prohibition",  # 9  No passing
    "prohibition",  # 10 No passing >3.5t
    "priority",  # 11 Right-of-way at the next intersection
    "priority",  # 12 Priority road
    "priority",  # 13 Yield
    "priority",  # 14 Stop
    "prohibition",  # 15 No vehicles
    "prohibition",  # 16 Vehicles >3.5t prohibited
    "prohibition",  # 17 No entry
    "warning",  # 18 General caution
    "warning",  # 19 Dangerous curve to the left
    "warning",  # 20 Dangerous curve to the right
    "warning",  # 21 Double curve
    "warning",  # 22 Bumpy road
    "warning",  # 23 Slippery road
    "warning",  # 24 Road narrows on the right
    "warning",  # 25 Road work
    "warning",  # 26 Traffic signals
    "warning",  # 27 Pedestrians
    "warning",  # 28 Children crossing
    "warning",  # 29 Bicycles crossing
    "warning",  # 30 Beware of ice/snow
    "warning",  # 31 Wild animals crossing
    "other",  # 32 End of all speed and passing limits
    "mandatory",  # 33 Turn right ahead
    "mandatory",  # 34 Turn left ahead
    "mandatory",  # 35 Ahead only
    "mandatory",  # 36 Go straight or right
    "mandatory",  # 37 Go straight or left
    "mandatory",  # 38 Keep right
    "mandatory",  # 39 Keep left
    "mandatory",  # 40 Roundabout mandatory
    "other",  # 41 End of no passing
    "other",  # 42 End of no passing >3.5t
)

CLASS_CATEGORY_NAMES: Final[tuple[str, ...]] = (
    "speed_limit",
    "prohibition",
    "priority",
    "warning",
    "mandatory",
    "other",
)

_ID_TO_NAME: Final[MappingProxyType[int, str]] = MappingProxyType(dict(enumerate(CLASS_NAMES)))
_ID_TO_SHORT_NAME: Final[MappingProxyType[int, str]] = MappingProxyType(
    dict(enumerate(CLASS_SHORT_NAMES))
)
_ID_TO_CATEGORY: Final[MappingProxyType[int, ClassCategory]] = MappingProxyType(
    dict(enumerate(CLASS_CATEGORIES))
)


def class_name(class_id: int) -> str:
    """Return the official name for ``class_id``.

    Raises:
        ValueError: if ``class_id`` is outside ``[0, 42]``. Failing loudly here is
            deliberate: a silent fallback would let a class-index mismatch reach
            an API response or an evaluation table.
    """
    try:
        return _ID_TO_NAME[class_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GTSRB class id {class_id!r}; expected an integer in [0, {NUM_CLASSES - 1}]"
        ) from exc


def class_short_name(class_id: int) -> str:
    """Return the compact name for ``class_id``."""
    try:
        return _ID_TO_SHORT_NAME[class_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GTSRB class id {class_id!r}; expected an integer in [0, {NUM_CLASSES - 1}]"
        ) from exc


def class_category(class_id: int) -> ClassCategory:
    """Return the sign family for ``class_id``."""
    try:
        return _ID_TO_CATEGORY[class_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GTSRB class id {class_id!r}; expected an integer in [0, {NUM_CLASSES - 1}]"
        ) from exc


def is_valid_class_id(class_id: int) -> bool:
    """Return ``True`` when ``class_id`` is a valid GTSRB class id."""
    return isinstance(class_id, int) and 0 <= class_id < NUM_CLASSES


def validate_label_space(labels: object) -> None:
    """Assert that ``labels`` matches the canonical class ordering.

    Checkpoints record the class names they were trained with; this is the check
    that stops a checkpoint trained under a different label mapping from being
    served.

    Raises:
        ValueError: if ``labels`` is not the canonical 43-name sequence.
    """
    as_tuple = tuple(labels) if isinstance(labels, (list, tuple)) else None
    if as_tuple is None:
        raise ValueError(f"Expected a list or tuple of class names, got {type(labels).__name__}")
    if len(as_tuple) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} class labels, got {len(as_tuple)}. "
            "The checkpoint's label space does not match this version of gtsrb."
        )
    if as_tuple != CLASS_NAMES:
        mismatches = [
            f"index {i}: {got!r} != {expected!r}"
            for i, (got, expected) in enumerate(zip(as_tuple, CLASS_NAMES, strict=True))
            if got != expected
        ]
        raise ValueError(
            "Checkpoint class labels do not match the canonical GTSRB ordering: "
            + "; ".join(mismatches[:5])
        )
