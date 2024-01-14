import requests
import pandas as pd


class ReplayBufferController:
    def __init__(self, batch_size):
        self.replay_buffer_host = "172.17.0.2"
        self.replay_buffer_port = 80
        self.batch_size = batch_size

    def sample_batch(self):
        data_dict = pd.read_json(requests.get(
            f"http://{self.replay_buffer_host}:{self.replay_buffer_port}/batch_data/{self.batch_size}").text).to_dict()
        return (
            list(data_dict['State'].values()), list(data_dict['Reward'].values()), list(data_dict['Action'].values()),
            list(data_dict['Next state'].values())
        )
