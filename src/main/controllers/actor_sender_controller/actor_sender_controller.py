import keras
import pika
import os


class ActorSenderController:
    def __init__(self, broker_host: str, routing_key: str):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(broker_host)
        )
        self.channel = self.connection.channel()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.routing_key = routing_key
        self.channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")

    def __send(self, routing_key, actor_model: keras.Model):
        file_path = f"{self.path}/resources/{routing_key}.keras"
        actor_model.save(file_path)
        with open(file_path, "rb") as actor_model_file:
            actor_model_bytes = actor_model_file.read()
        self.channel.basic_publish(
            exchange="topic_exchange", routing_key=routing_key, body=actor_model_bytes
        )

    def send_actors(self, actor_model):
        """
        Send actor model to Predator Prey service
        :param actor_model: model
        :return:
        """
        self.__send(self.routing_key, actor_model)
