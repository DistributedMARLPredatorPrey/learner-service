from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Sum type modelling the environment configuration.
    """

    num_predators: int
    num_preys: int
    num_states: int


@dataclass(frozen=True)
class ReplayBufferServiceConfig:
    """
    Sum type modelling the replay buffer configuration.
    """

    project_root_path: str
    batch_size: int
    replay_buffer_host: str
    replay_buffer_port: int


@dataclass
class LearnerServiceConfig:
    """
    Sum type modelling the lerner service configuration.
    """

    pubsub_broker: str
