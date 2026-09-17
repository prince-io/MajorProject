"""Unified 6-class mapping for ACDC and BDD100K.

This module is a pure, side-effect-free mapping layer: importing it must not read
files, load data, or mutate global state beyond defining the constants below.

Unified scheme (6 classes)::

    0 person, 1 rider, 2 car, 3 truck, 4 bus, 5 bicycle (motorcycle folded in)

Rationale: ACDC annotates 8 categories and BDD100K annotates 10. The road-user
intersection we care about is person/rider/car/truck/bus/bicycle. Motorcycles are
folded into ``bicycle``; ``train``, ``traffic light``, and ``traffic sign`` are
dropped.
"""

from __future__ import annotations

UNIFIED_CLASSES: dict[int, str] = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "bicycle",  # includes motorcycle (folded in)
}
"""Canonical unified class IDs to their human-readable names."""

BDD_MAP: dict[str, int] = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "bike": 5,
    "motor": 5,  # motorcycle -> bicycle
    # dropped: "traffic light", "traffic sign", "train"
}
"""BDD100K native category string to unified class ID."""

ACDC_MAP: dict[int, int | None] = {
    24: 0,  # person
    25: 1,  # rider
    26: 2,  # car
    27: 3,  # truck
    28: 4,  # bus
    31: None,  # train -- drop
    32: 5,  # motorcycle -> bicycle
    33: 5,  # bicycle
}
"""ACDC COCO category ID to unified class ID (``None`` means drop)."""


def bdd_category_to_unified(cat: str) -> int | None:
    """Map a BDD100K native category name to a unified class ID.

    Args:
        cat: BDD100K category string, e.g. ``"bike"`` or ``"traffic light"``.

    Returns:
        The unified class ID, or ``None`` if the category is not part of the
        unified scheme and should be dropped.
    """
    return BDD_MAP.get(cat)


def acdc_category_to_unified(cat_id: int) -> int | None:
    """Map an ACDC COCO category ID to a unified class ID.

    Args:
        cat_id: ACDC COCO category ID, e.g. ``24`` (person) or ``31`` (train).

    Returns:
        The unified class ID, or ``None`` if the category is not part of the
        unified scheme and should be dropped.
    """
    return ACDC_MAP.get(cat_id)


def _print_mapping(title: str, mapping: dict[object, object]) -> None:
    """Print a single mapping table under a titled header."""
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in mapping.items():
        if value is None:
            rendered = "DROPPED"
        elif isinstance(value, int) and value in UNIFIED_CLASSES:
            rendered = f"{value} ({UNIFIED_CLASSES[value]})"
        else:
            rendered = str(value)
        print(f"  {key!r:>18} -> {rendered}")


if __name__ == "__main__":
    print("=" * 60)
    print("Unified class scheme")
    print("=" * 60)
    for class_id, name in UNIFIED_CLASSES.items():
        print(f"  {class_id}: {name}")

    _print_mapping("BDD100K category string -> unified ID", BDD_MAP)
    _print_mapping("ACDC COCO category ID -> unified ID", ACDC_MAP)
