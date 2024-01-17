import os
import yaml

from src.main.controllers.parser.conf import EnvironmentConf, LearnerConf


class YamlConfParser:

    def __init__(self):
        conf_path = os.environ.get('CONF_PATH')
        with open(conf_path, 'r') as conf:
            self.configurations = yaml.safe_load(conf)

    def environment_configuration(self):
        env_conf = self.configurations["environment"]
        return EnvironmentConf(
            num_predators=env_conf["num_predators"],
            num_preys=env_conf["num_preys"],
            acc_lower_bound=env_conf["acc_lower_bound"],
            acc_upper_bound=env_conf["acc_upper_bound"],
            num_states=env_conf["num_states"],
            num_actions=env_conf["num_actions"]
        )

    def learner_configuration(self):
        learner_conf = self.configurations["learner"]
        return LearnerConf(
            batch_size=learner_conf["batch_size"],
            batch_url=os.environ.get('BATCH_SUBDIR')
        )
