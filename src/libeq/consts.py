from enum import IntEnum, auto
import math
from typing import Final

class Flags(IntEnum):
    CONSTANT = auto()
    REFINE = auto()

LN10: Final[float] = math.log(10)
