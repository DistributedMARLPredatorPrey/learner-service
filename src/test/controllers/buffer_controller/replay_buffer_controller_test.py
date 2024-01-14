import unittest

from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)


class ReplayBufferControllerTest(unittest.TestCase):
    batch_size = 10
    replay_buffer_controller: ReplayBufferController = ReplayBufferController(
        batch_size
    )

    def test_batch_data(self):
        (
            states,
            rewards,
            actions,
            next_states,
        ) = self.replay_buffer_controller.sample_batch()
        self.assertEqual(len(states), self.batch_size)
        self.assertEqual(len(rewards), self.batch_size)
        self.assertEqual(len(actions), self.batch_size)
        self.assertEqual(len(next_states), self.batch_size)
