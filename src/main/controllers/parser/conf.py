from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConf:
    num_predators: int
    num_preys: int
    acc_lower_bound: float
    acc_upper_bound: float
    num_states: int
    num_actions: int


@dataclass(frozen=True)
class LearnerConf:
    batch_size: int
    batch_url: str
