import pika


class ActorSenderController:

    def __init__(self):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

        # Declare a topic exchange named 'topic_exchange'
        self.channel.exchange_declare(exchange='topic_exchange', exchange_type='topic')

    def _send(self, routing_key, actor_model):
        # Publish the message to the topic exchange with the specified routing key
        self.channel.basic_publish(exchange='topic_exchange',
                                   routing_key=routing_key,
                                   body=actor_model)

        print(f" [x] Sent '{actor_model}' with routing key '{routing_key}'")

    def send_actors(self, actor_models):
        for i in range(len(actor_models)):
            self._send(f"actor-model-{i}", actor_models[i])
