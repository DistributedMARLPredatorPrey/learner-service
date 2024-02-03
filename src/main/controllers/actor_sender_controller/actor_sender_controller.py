import pika


class ActorSenderController:
    def __init__(self, broker_host: str):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(broker_host)
        )
        self.channel = self.connection.channel()

        # Declare a topic exchange named 'topic_exchange'
        self.channel.exchange_declare(exchange="topic_exchange", exchange_type="topic")

    def _send(self, routing_key, actor_model):

        file_path = f"resources/{routing_key}.keras"
        actor_model.save(file_path)

        with open(file_path, 'rb') as actor_model_file:
            # Publish the message to the topic exchange with the specified routing key
            self.channel.basic_publish(
                exchange="topic_exchange", routing_key=routing_key, body=actor_model_file
            )

        print(f" [x] Sent '{actor_model}' with routing key '{routing_key}'")

    def send_actors(self, actor_models):
        for i in range(len(actor_models)):
            self._send(f"actor-model-{i}", actor_models[i])
