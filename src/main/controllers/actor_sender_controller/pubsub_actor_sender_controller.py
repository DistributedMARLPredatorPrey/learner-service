from typing import Tuple

import keras
import pika
import os

from src.main.controllers.actor_sender_controller.actor_sender_controller import (
    ActorSenderController,
)
from src.main.model.actor_critic.actor import Actor


class PubSubActorSenderController(ActorSenderController):
    def __init__(
        self,
        project_root_path: str,
        broker_host: str,
        predator_routing_key: str,
        prey_routing_key: str,
    ):
        self.__broker_host = broker_host
        self.__path = os.path.join(project_root_path, "src", "main", "resources")
        self.__predator_routing_key = predator_routing_key
        self.__prey_routing_key = prey_routing_key
        self.__connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.__broker_host, heartbeat=600)
        )
        self.__channel = self.__connection.channel()
        self.__channel.exchange_declare(
            exchange="topic_exchange", exchange_type="topic"
        )

    def __send(self, routing_key: str, actor_model: keras.Model):
        file_path = os.path.join(self.__path, f"{routing_key}.keras")
        actor_model.save(file_path)
        with open(file_path, "rb") as actor_model_file:
            actor_model_bytes = actor_model_file.read()
        self.__channel.basic_publish(
            exchange="topic_exchange", routing_key=routing_key, body=actor_model_bytes
        )

    def send_actors(self, predator_actor: Actor, prey_actor: Actor):
        """
        Send actor model to Predator Prey service using the PubSub channel
        :param predator_actor: Predator Actor
        :param prey_actor: Prey Actor
        :return:
        """
        self.__send(self.__predator_routing_key, predator_actor)
        self.__send(self.__prey_routing_key, prey_actor)
