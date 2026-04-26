from enum import Enum


class MatrixType(str, Enum):
    """Type of scattering sub-matrix to compute."""

    TM = "TM"
    """Transmission matrix (output ports)."""

    RM = "RM"
    """Reflection matrix (reflection ports)."""
