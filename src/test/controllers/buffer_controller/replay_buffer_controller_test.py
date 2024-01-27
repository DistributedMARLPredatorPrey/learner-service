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

# class BufferTest(unittest.TestCase):
#     batch_size, num_states, num_actions, num_agents = 3, 2, 1, 2
#
#     buffer = Buffer(
#         batch_size=batch_size,
#         num_states=num_states,
#         num_actions=num_actions,
#         num_agents=num_agents,
#     )
#
#     def test_batch_size(self):
#         self.buffer.record(([0, 0, 0, 0], [0, 0], [0, 0], [0, 0, 0, 0]))
#         s, a, r, ns = self.buffer.sample_batch()
#         assert all(
#             [
#                 np.array(s).shape
#                 == (self.batch_size, self.num_states * self.num_agents),
#                 np.array(a).shape
#                 == (self.batch_size, self.num_actions * self.num_agents),
#                 np.array(r).shape == (self.batch_size, self.num_agents),
#                 np.array(ns).shape
#                 == (self.batch_size, self.num_states * self.num_agents),
#             ]
#         )
