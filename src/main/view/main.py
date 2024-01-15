from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)

from src.main.controllers.learner_controller.learner import Learner
from src.main.controllers.parser.conf import EnvironmentConf, LearnerConf
from src.main.controllers.parser.yaml_conf_parser import YamlConfParser

if __name__ == "__main__":

    conf: YamlConfParser = YamlConfParser()
    env_conf: EnvironmentConf = conf.environment_configuration()
    learner_conf: LearnerConf = conf.learner_configuration()

    learner = Learner(
        replay_buffer_controller=ReplayBufferController(
            batch_size=learner_conf.batch_size,
            num_states=env_conf.num_states,
            num_actions=env_conf.num_actions,
            num_agents=env_conf.num_predators,
        ),
        num_agents=env_conf.n,
        num_states=num_states,
        num_actions=num_actions,
    )

    while True:
        learner.update()
