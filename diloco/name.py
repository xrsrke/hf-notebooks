from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class Datatype(Enum):
    FLOAT32 = auto()
    BFLOAT16 = auto()
    FP8 = auto()
    INT4 = auto()


class GPU(Enum):
    H100 = auto()

@dataclass
class Supercomputer:
    name: str
    coordinate: tuple

@dataclass
class Transformer:
    name: str
    n_layers: int
    hidden_size: int
    n_heads: int
    n_key_value_heads: int

@dataclass
class TrainingConfig:
    tp_size: int
    pp_size: int
    num_gpus: int
    ctx_length: int
    partition_activations: bool
    zero1: bool
    checkpoint_activations: bool
    batch_size_per_replicas: int
    weight_dtype: Datatype
    act_dtype: Datatype
    gradient_dtype: Datatype

    optim_first_state_dtype: Datatype
    optim_second_state_dtype: Datatype
    master_weight_dtype: Optional[Datatype] = None

    def __post_init__(self):
        assert self.num_gpus // (self.tp_size * self.pp_size)
        self.dp_size = self.num_gpus // (self.tp_size * self.pp_size)
        assert self.act_dtype == Datatype.BFLOAT16
