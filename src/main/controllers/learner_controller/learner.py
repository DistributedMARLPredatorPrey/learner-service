from time import sleep

import numpy as np
import tensorflow as tf

from src.main.controllers.actor_sender_controller.actor_sender_controller import (
    ActorSenderController,
)
from src.main.controllers.buffer_controller.replay_buffer_controller import (
    ReplayBufferController,
)
from src.main.model.actor_critic.actor import Actor
from src.main.model.actor_critic.critic import Critic


class Learner:
    def __init__(
        self,
        replay_buffer_controller: ReplayBufferController,
        num_predators: int,
        num_preys: int,
        num_states: int,
        actor_sender_controller: ActorSenderController,
    ):
        """
        Initializes a Learner.
        A Learner updates each agent Actor-Critic network, where the agents are from the same type,
        by batching data from a shared buffer.
        In particular, the learning follows the MADDPG algorithm.
        Moreover, it sets the latest actor network to each agent's ParameterService, through which each
        agent will take an action given a state.
        :param replay_buffer_controller: shared buffer
        :param num_states: state size
        :param num_actions: number of actions allowed
        :param num_agents: number of agents of the same type
        """
        self.replay_buffer_controller = replay_buffer_controller
        self.num_states = num_states
        self.num_actions = 2
        self.num_agents = num_predators + num_preys

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.num_predators = num_predators
        self.num_preys = num_preys

        # Creating critic models
        self.critic_models, self.target_critics = [], []
        self.critic_optimizers = []
        for i in range(2):
            critic_model, target_critic = (
                Critic(num_states, self.num_actions, self.num_agents).model,
                Critic(num_states, self.num_actions, self.num_agents).model,
            )
            target_critic.set_weights(critic_model.get_weights())
            critic_opt = tf.keras.optimizers.Adam(self.critic_lr)
            self.critic_models.append(critic_model)
            self.target_critics.append(target_critic)
            self.critic_optimizers.append(critic_opt)

        # Creating target actor model
        self.actor_models, self.target_actors = [], []
        self.actor_optimizers = []
        for i in range(2):
            actor_model, target_actor = Actor(num_states).model, Actor(num_states).model
            target_actor.set_weights(actor_model.get_weights())
            actor_opt = tf.keras.optimizers.Adam(self.actor_lr)
            self.actor_models.append(actor_model)
            self.target_actors.append(target_actor)
            self.actor_optimizers.append(actor_opt)

        self.actor_sender_controller = actor_sender_controller
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005
        # Send initial actor without training
        self.actor_sender_controller.send_actors(
            (self.actor_models[0], self.actor_models[1])
        )

    def update(self):
        """
        Updates the Actor-Critic network of each agent, following the MADDPG algorithm.
        :return:
        """
        self.__update_actors(self.__update_critics())
        self.__update_targets()
        self.actor_sender_controller.send_actors(
            (self.actor_models[0], self.actor_models[1])
        )

    def __update_targets(self):
        """
        Slowly updates target parameters according to the tau rate <<1
        :return:
        """
        for i in range(2):
            self.__update_target(self.target_actors[i], self.actor_models[i], self.tau)
            self.__update_target(
                self.target_critics[i], self.critic_models[i], self.tau
            )

    def __update_target(self, target, original, tau):
        target_weights = target.get_weights()
        original_weights = original.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = original_weights[i] * tau + target_weights[i] * (
                1 - tau
            )
        target.set_weights(target_weights)

    def __update_critics(self):
        """
        Updates the Critic networks by reshaping the sampled data.
        :return:
        """
        batch_timeout = 30
        # Batch a sample from the buffer waiting one second between the requests
        # if there isn't any yet recorded
        data_as_tuple = self.replay_buffer_controller.sample_batch()
        while data_as_tuple is None:
            sleep(batch_timeout)
            data_as_tuple = self.replay_buffer_controller.sample_batch()

        (state_batch, action_batch, reward_batch, next_state_batch) = data_as_tuple
        target_actions = []

        for j in range(self.num_agents):
            target_actions.append(
                self.target_actors[j > self.num_predators](
                    # get the next state of the j-agent
                    next_state_batch[
                        :, j * self.num_states : (j + 1) * self.num_states
                    ],
                    training=True,
                )
            )
        action_batch_reshape = []
        for j in range(self.num_agents):
            action_batch_reshape.append(
                action_batch[:, j * self.num_actions : (j + 1) * self.num_actions]
            )

        state_batch, losses = self.__update_critic_networks(
            state_batch,
            reward_batch,
            action_batch_reshape,
            next_state_batch,
            target_actions,
        )

        predator_losses, prey_losses = losses[0], losses[1]
        with open("predator_losses.txt", "a") as f:
            for loss in predator_losses:
                f.write(str(loss.numpy()) + "\n")
        with open("prey_losses.txt", "a") as f:
            for loss in prey_losses:
                f.write(str(loss.numpy()) + "\n")

        return state_batch

    @tf.function
    def __update_critic_networks(
        self, state_batch, reward_batch, action_batch, next_state_batch, target_actions
    ):
        """
        Computes the loss and updates parameters of the Critic networks.
        Makes use of tensorflow graphs to speed up the computation.
        :param state_batch: state batch
        :param reward_batch: reward batch
        :param action_batch: action batch
        :param next_state_batch: next state batch
        :param target_actions: target actions
        :return:
        """
        map_arr = lambda arr, i, l: tf.concat(
            [arr[:, i * l : (i + 1) * l], arr[:, : i * l], arr[:, (i + 1) * l :]],
            axis=1,
        )
        stack = lambda arr: tf.stack(arr)
        unstack = lambda arr: tf.unstack(arr)

        losses = [[], []]
        for i in range(self.num_agents):
            # Train the Critic network
            model_i = int(i > (self.num_predators - 1))
            with tf.GradientTape() as tape:
                # Compute y = r_i + gamma * Q_i'(x',a_1',a_2', ...,a_n')
                y = reward_batch[:, i] + self.gamma * self.target_critics[model_i](
                    [
                        map_arr(next_state_batch, i, self.num_states),
                        unstack(map_arr(stack(target_actions), i, self.num_actions)),
                    ],
                    training=True,
                )
                # Eval Q_i(x, a_1, a_2, ..., a_n)
                critic_value = self.critic_models[model_i](
                    [
                        map_arr(state_batch, i, self.num_states),
                        unstack(map_arr(stack(action_batch), i, self.num_actions)),
                    ],
                    training=True,
                )
                # Compute loss = square_mean(y - Q_i(x, a_1, a_2, ..., a_n))
                # Here square_mean is taken over the batch
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
                losses[model_i].append(critic_loss)

            critic_grad = tape.gradient(
                critic_loss, self.critic_models[model_i].trainable_variables
            )
            self.critic_optimizers[model_i].apply_gradients(
                zip(critic_grad, self.critic_models[model_i].trainable_variables)
            )

        return state_batch, losses

    def __update_actors(self, state_batch):
        """
        Updates the Actor networks by:
        - Computing the loss from the Q-value of each agent Critic
        - Applying gradient to the Actor network
        :param state_batch: state batch
        :return:
        """
        actions = []
        for i in range(self.num_agents):
            actions.append(
                self.actor_models[i > self.num_predators - 1](
                    state_batch[:, i * self.num_states : (i + 1) * self.num_states],
                    training=True,
                )
            )
        self.__update_actor_networks(state_batch, actions)

    @tf.function
    def __update_actor_networks(self, state_batch, actions):
        """
        Computes the loss and updates parameters of the Actor networks.
        Makes use of tensorflow graphs to speed up the computation.
        :param state_batch: state batch
        :param actions: joint actions
        :return:
        """
        map_arr = lambda arr, i, l: tf.concat(
            [arr[:, i * l : (i + 1) * l], arr[:, : i * l], arr[:, (i + 1) * l :]],
            axis=1,
        )
        stack = lambda arr: tf.stack(arr)
        unstack = lambda arr: tf.unstack(arr)

        for i in range(self.num_agents):
            model_i = int(i > (self.num_predators - 1))

            with tf.GradientTape(persistent=True) as tape:
                action = self.actor_models[model_i](
                    [state_batch[:, i * self.num_states : (i + 1) * self.num_states]],
                    training=True,
                )
                critic_value = self.critic_models[model_i](
                    [
                        map_arr(state_batch, i, self.num_states),
                        unstack(
                            map_arr(
                                stack(
                                    [
                                        actions[k] if k != i else action
                                        for k in range(self.num_agents)
                                    ]
                                ),
                                i,
                                self.num_actions,
                            )
                        ),
                    ],
                    training=True,
                )
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(
                actor_loss, self.actor_models[model_i].trainable_variables
            )
            self.actor_optimizers[model_i].apply_gradients(
                zip(actor_grad, self.actor_models[model_i].trainable_variables)
            )
