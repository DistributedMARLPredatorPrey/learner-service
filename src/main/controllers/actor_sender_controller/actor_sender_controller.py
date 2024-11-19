from typing import Tuple

from src.main.model.actor_critic.actor import Actor


class ActorSenderController:
    def send_actors(self, predator_actor: Actor, prey_actor: Actor):
        """
        Send actor model to Predator Prey service
        :param predator_actor: Predator Actor
        :param prey_actor: Prey Actor
        :return:
        """
        raise NotImplementedError("Subclasses must implement this method")
