import unittest

from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.controllers.config_utils.config import ReplayBufferServiceConfig


# class ReplayBufferControllerTest(unittest.TestCase):
#     batch_size = 10
#     replay_buffer_controller: ReplayBufferController = ReplayBufferController(
#         ReplayBufferServiceConfig(
#             batch_size=batch_size,
#             agent_type="predator",
#             replay_buffer_host="172.18.0.2",
#             replay_buffer_port=80,
#         )
#     )
#
#     def test_batch_data(self):
#         (
#             states,
#             rewards,
#             actions,
#             next_states,
#         ) = self.replay_buffer_controller.sample_batch()
#         self.assertEqual(len(states), self.batch_size)
#         self.assertEqual(len(rewards), self.batch_size)
#         self.assertEqual(len(actions), self.batch_size)
#         self.assertEqual(len(next_states), self.batch_size)
