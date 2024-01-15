from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)

from src.main.controllers.learner_controller.learner import Learner

if __name__ == "__main__":
    batch_size = 64
    num_states = 2
    num_actions = 2
    num_agents = 10

    learner = Learner(
        replay_buffer_controller=ReplayBufferController(
            batch_size,
            num_states=num_states,
            num_actions=num_actions,
            num_agents=num_agents,
        ),
        num_agents=num_agents,
        num_states=num_states,
        num_actions=num_actions,
    )

    while True:
        learner.update()
