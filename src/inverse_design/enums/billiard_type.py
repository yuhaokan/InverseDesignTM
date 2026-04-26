from enum import Enum


class BilliardType(str, Enum):
    """Billiard cavity configurations."""

    TWO_PORT = "BilliardTwo"
    """2-port billiard cavity (2x2 transmission matrix)."""

    THREE_PORT = "BilliardThree"
    """3-port billiard cavity (3x3 transmission matrix)."""
