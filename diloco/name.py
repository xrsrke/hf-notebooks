from enum import Enum, auto


class Datatype(Enum):
    FP8 = auto()
    BFLOAT16 = auto()


class GPU(Enum):
    H100 = auto()
