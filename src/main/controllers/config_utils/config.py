from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Sum type modelling the environment configuration.
    """

    num_predators: int
    num_preys: int
    acc_lower_bound: float
    acc_upper_bound: float
    num_states: int
    num_actions: int


@dataclass(frozen=True)
class ReplayBufferServiceConfig:
    """
    Sum type modelling the replay buffer configuration.
    """

    batch_size: int
    agent_type: str
    replay_buffer_host: str
    replay_buffer_port: str
