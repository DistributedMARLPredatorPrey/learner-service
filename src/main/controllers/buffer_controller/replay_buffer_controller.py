from typing import Dict, Tuple

import requests
import pandas as pd
import tensorflow as tf
from requests import Response


class ReplayBufferController:
    def __init__(self, batch_size, num_states, num_actions, num_agents):
        self.replay_buffer_host = "172.17.0.2"
        self.replay_buffer_port = 80
        self.batch_size = batch_size
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = num_agents

    def sample_batch(self) -> tuple:
        """
        Sample a data batch of batch_size
        :return: (state_batch, action_batch, reward_batch, next_state_batch) tuple
        """
        return self._convert_data_batch(
            pd.DataFrame(self._request_data_batch().json()).to_dict()
        )

    def _request_data_batch(self) -> Response:
        """
        Gets a data batch from a distributed Replay buffer service.
        :return: Response from the remote server
        """
        return requests.get(
            f"http://{self.replay_buffer_host}:{self.replay_buffer_port}/batch_data/{self.batch_size}"
        )

    @staticmethod
    def _convert_data_batch(data_dict: Dict) -> Tuple:
        """
        Converts a data batch of type Dict to a Tuple
        :param data_dict:
        :return:
        """
        return tuple(
            [
                tf.convert_to_tensor(
                    [
                        [float(vv) for vv in v.split("|")]
                        for v in list(data_dict[column].values())
                    ]
                )
                for column in ["State", "Action", "Reward", "Next state"]
            ]
        )
