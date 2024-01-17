from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)

from src.main.controllers.learner_controller.learner import Learner
from src.main.controllers.config_utils.config import EnvironmentConfig, LearnerConfig
from src.main.controllers.config_utils.config_utils import ConfigUtils

if __name__ == "__main__":
    config_utils: ConfigUtils = ConfigUtils()
    env_config: EnvironmentConfig = config_utils.environment_configuration()
    learner_config: LearnerConfig = config_utils.learner_configuration()

    # Get the number of agents based whether the type is predator or prey
    num_agents = (
        env_config.num_predators
        if learner_config.agent_type == "predator"
        else env_config.num_preys
    )
    # Create the learner, passing to it a replay buffer controller
    learner = Learner(
        replay_buffer_controller=ReplayBufferController(
            batch_size=learner_config.batch_size,
            num_states=env_config.num_states,
            num_actions=env_config.num_actions,
            num_agents=num_agents,
        ),
        num_agents=num_agents,
        num_states=env_config.num_states,
        num_actions=env_config.num_actions,
    )
    # Train the models until the process is being stopped
    while True:
        learner.update()
