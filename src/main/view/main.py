import logging
import time

from src.main.controllers.actor_sender_controller.actor_sender_controller import (
    ActorSenderController,
)
from src.main.controllers.actor_sender_controller.pubsub_actor_sender_controller import (
    PubSubActorSenderController,
)
from src.main.controllers.buffer_controller.remote_replay_buffer_controller import (
    RemoteReplayBufferController,
)
from src.main.controllers.learner_controller.learner import Learner

from src.main.controllers.learner_controller.maddpg.maddpg_learner import MADDPGLearner
from src.main.model.config.config import (
    EnvironmentConfig,
    ReplayBufferServiceConfig,
)
from src.main.model.config.config_utils import ConfigUtils

logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    # Create configuration
    config_utils: ConfigUtils = ConfigUtils()
    env_config: EnvironmentConfig = config_utils.environment_configuration()
    replay_buffer_service_config: ReplayBufferServiceConfig = (
        config_utils.replay_buffer_configuration()
    )
    logging.info("Starting Learner Service...")
    actor_sender_controller: ActorSenderController = PubSubActorSenderController(
        project_root_path=config_utils.replay_buffer_configuration().project_root_path,
        broker_host=config_utils.learner_service_configuration().pubsub_broker,
        predator_routing_key="predator-actor-model",
        prey_routing_key="prey-actor-model",
    )
    learner: Learner = MADDPGLearner(
        replay_buffer_controller=RemoteReplayBufferController(
            replay_buffer_service_config
        ),
        num_predators=env_config.num_predators,
        num_preys=env_config.num_preys,
        num_states=env_config.num_states,
        root_project_path=replay_buffer_service_config.project_root_path,
        actor_sender_controller=actor_sender_controller,
    )
    # Train the models until the process is being stopped
    DT = 30
    while True:
        predator_actor, prey_actor = learner.update()
        actor_sender_controller.send_actors(
            predator_actor=predator_actor, prey_actor=prey_actor
        )
        time.sleep(DT)
