from enum import Enum, auto
from dataclasses import dataclass


class Datatype(Enum):
    FP8 = auto()
    BFLOAT16 = auto()


class GPU(Enum):
    H100 = auto()

@dataclass
class Supercomputer:
    name: str
    coordinate: tuple
