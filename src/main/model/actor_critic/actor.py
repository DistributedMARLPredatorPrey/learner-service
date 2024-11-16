import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import layers


class Actor:
    def __init__(self, num_states: int):
        """
        The Actor network is responsible to choose the action, given the state.
        The action is a tuple (velocity, angular_velocity).
        :param num_states: number of states
        """
        last_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
        inputs = layers.Input(shape=(num_states,))
        inner = layers.Dense(128, activation="relu")(inputs)
        inner = layers.Dense(64, activation="relu")(inner)
        # velocity
        vel_out = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(
            inner
        )
        # angular velocity
        ang_vel_out = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(
            inner
        )
        outputs = layers.Concatenate()([vel_out, ang_vel_out])
        self.model = tf.keras.Model(inputs, outputs, name="actor")
