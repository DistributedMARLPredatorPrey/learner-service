from src.main.controllers.actor_sender_controller.actor_sender_controller import (
    ActorSenderController,
)
from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)

from src.main.controllers.learner_controller.learner import Learner
from src.main.controllers.config_utils.config import (
    EnvironmentConfig,
    ReplayBufferServiceConfig,
)
from src.main.controllers.config_utils.config_utils import ConfigUtils

if __name__ == "__main__":
    config_utils: ConfigUtils = ConfigUtils()
    env_config: EnvironmentConfig = config_utils.environment_configuration()
    replay_buffer_service_config: ReplayBufferServiceConfig = (
        config_utils.replay_buffer_configuration()
    )

    # Create the learner, passing to it a replay buffer controller
    learner = Learner(
        replay_buffer_controller=ReplayBufferController(replay_buffer_service_config),
        num_agents=(
            # Get the number of agents based whether the type is predator or prey
            env_config.num_predators
            if replay_buffer_service_config.agent_type == "predator"
            else env_config.num_preys
        ),
        num_states=env_config.num_states,
        num_actions=env_config.num_actions,
        actor_sender_controller=ActorSenderController(
            config_utils.learner_service_configuration().pubsub_broker
        ),
    )
    # Train the models until the process is being stopped
    while True:
        learner.update()
