from enum import Enum


class TargetType(str, Enum):
    """Reward objectives used by billiard environments."""

    RANK1 = "Rank1"
    RANK1_TRACE0 = "Rank1Trace0"
    DEGENERATE_EIG_VAL = "DegenerateEigVal"
    FIXED_TARGET = "FixedTarget"
    DEGENERATE_SINGULAR_VAL = "DegenerateSingularVal"
