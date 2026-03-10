from enum import IntEnum, auto
import math
from typing import Final

class Flags(IntEnum):
    """Refinement flags used to mark whether a parameter is fixed or to be optimized.

    Attributes
    ----------
    CONSTANT : int
        The parameter is held constant and will not be refined.
    REFINE : int
        The parameter is free to be refined during optimization.
    """

    CONSTANT = auto()
    REFINE = auto()

LN10: Final[float] = math.log(10)
