from tensorflow.keras import Model


class ActorModelStore:
    def __init__(self):
        self.actor_model = None

    def set_model(self, actor_model: Model):
        self.actor_model = actor_model
