from typing import Tuple

import keras
import pika
import os

from src.main.model.actor_critic.actor import Actor


class ActorSenderController:
    def __init__(self, broker_host: str, routing_keys: Tuple[str, str]):
        self.broker_host = broker_host
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.routing_keys = routing_keys
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(self.broker_host)
        )
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")

    def __send(self, routing_key, actor_model: keras.Model):
        file_path = f"{self.path}/resources/{routing_key}.keras"
        actor_model.save(file_path)
        with open(file_path, "rb") as actor_model_file:
            actor_model_bytes = actor_model_file.read()
        self.channel.basic_publish(
            exchange="topic_exchange", routing_key=routing_key, body=actor_model_bytes
        )

    def send_actors(self, actor_models: Tuple[Actor, Actor]):
        """
        Send actor model to Predator Prey service
        :param actor_models: models
        :return:
        """
        self.__send(self.routing_keys[0], actor_models[0])
        self.__send(self.routing_keys[1], actor_models[1])
