import ast
import logging
from typing import Dict, Tuple

import requests
import pandas as pd
import tensorflow as tf
from requests import Response

from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.model.config.config import ReplayBufferServiceConfig


class RemoteReplayBufferController(ReplayBufferController):
    def __init__(self, replay_buffer_service_config: ReplayBufferServiceConfig):
        self.__replay_buffer_host = replay_buffer_service_config.replay_buffer_host
        self.__replay_buffer_port = replay_buffer_service_config.replay_buffer_port
        self.__batch_size = replay_buffer_service_config.batch_size

    def sample_batch(self) -> tuple:
        """
        Sample a data batch of batch_size
        :return: (state_batch, action_batch, reward_batch, next_state_batch) tuple
        """
        df_data = pd.DataFrame(self.__request_data_batch().json())
        return None if df_data.empty else self.__convert_data_batch(df_data.to_dict())

    def __request_data_batch(self) -> Response:
        """
        Gets a data batch from a distributed Replay buffer service.
        :return: Response from the remote server
        """
        return requests.get(
            f"http://{self.__replay_buffer_host}:{self.__replay_buffer_port}/"
            f"batch_data/{self.__batch_size}"
        )

    @staticmethod
    def __convert_data_batch(data_dict: Dict) -> Tuple:
        """
        Converts a data batch of type Dict to a Tuple
        :param data_dict: Data batch as a dict
        :return: Data batch as a tuple of tf Tensors
        """
        return tuple(
            [
                tf.convert_to_tensor(
                    [
                        ast.literal_eval(vv) if isinstance(vv, str) else vv
                        for vv in list(data_dict[c].values())
                    ],
                    dtype=tf.float32,
                )
                for c in ["State", "Action", "Reward", "Next State"]
            ]
        )
