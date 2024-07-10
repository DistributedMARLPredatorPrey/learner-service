import logging
import time

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
    # Create configuration
    config_utils: ConfigUtils = ConfigUtils()
    env_config: EnvironmentConfig = config_utils.environment_configuration()
    replay_buffer_service_config: ReplayBufferServiceConfig = (
        config_utils.replay_buffer_configuration()
    )

    # Create the learner, passing to it a replay buffer controller
    logging.info("Starting Learner...")
    learner = Learner(
        replay_buffer_controller=ReplayBufferController(replay_buffer_service_config),
        agent_type=replay_buffer_service_config.agent_type,
        num_predators=env_config.num_predators,
        num_preys=env_config.num_preys,
        num_states=env_config.num_states,
        num_actions=env_config.num_actions,
        save_path=replay_buffer_service_config.agent_type,
        actor_sender_controller=ActorSenderController(
            config_utils.learner_service_configuration().pubsub_broker,
            "predator-actor-model"
            if replay_buffer_service_config.agent_type == "predator"
            else "prey-actor-model",
        ),
    )
    # Train the models until the process is being stopped
    t_step = 1
    while True:
        learner.update()
        time.sleep(t_step)
