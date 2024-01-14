from src.main.controllers.learner_controller.learner import Learner


class LearnerFactory:
    @staticmethod
    def create_learners(
        replay_buffer_controllers, par_services, num_states, num_actions
    ):
        """
        Create a new Learner for each buffer passed as parameter
        :param replay_buffer_controllers: Replay buffer controllers
        :param par_services: ParameterServices
        :param num_states: number of states
        :param num_actions: number of actions
        :return:
        """
        return [
            Learner(
                replay_buffer_controllers[i],
                par_services[i],
                num_states,
                num_actions,
                len(par_services[i]),
            )
            for i in range(len(replay_buffer_controllers))
        ]
