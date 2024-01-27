import pika


class ActorSenderController:

    def __init__(self, routing_key: str):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        self.channel = self.connection.channel()

        # Declare a topic exchange named 'topic_exchange'
        self.channel.exchange_declare(exchange='topic_exchange', exchange_type='topic')
        self.routing_key = routing_key

    def send(self, actor_model):
        # Publish the message to the topic exchange with the specified routing key
        self.channel.basic_publish(exchange='topic_exchange',
                                   routing_key=self.routing_key,
                                   body=actor_model)

        print(f" [x] Sent '{actor_model}' with routing key '{self.routing_key}'")


sender = ActorSenderController("actor-model")
sender.send("Ciaoooo")
