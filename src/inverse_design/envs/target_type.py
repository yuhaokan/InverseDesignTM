from enum import Enum


class TargetType(str, Enum):
    """Reward objectives used by billiard environments.

    Each value selects a different error function for the transmission matrix.
    Lower error means higher reward.
    """

    RANK1 = "Rank1"
    """Minimize non-dominant singular content so TM behaves like rank-1."""

    RANK1_TRACE0 = "Rank1Trace0"
    """Promote rank-1 behavior while also driving matrix trace toward zero."""

    DEGENERATE_EIG_VAL = "DegenerateEigVal"
    """Drive eigenvalue discriminant toward zero (coalescing eigenvalues)."""

    FIXED_TARGET = "FixedTarget"
    """Match TM to a predefined complex target matrix."""

    DEGENERATE_SINGULAR_VAL = "DegenerateSingularVal"
    """Make leading singular values equal (singular-value degeneracy)."""
