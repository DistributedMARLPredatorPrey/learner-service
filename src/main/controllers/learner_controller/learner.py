import logging
from time import sleep

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
            agent_type: str,
            num_predators: int,
            num_preys: int,
            num_states: int,
            num_actions: int,
            save_path: str,
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
        # Parameters
        self.replay_buffer_controller = replay_buffer_controller
        # self.par_services = par_services
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_agents = num_predators + num_preys

        start_idx = 0 if agent_type == "predator" else num_predators
        end_idx = num_predators if agent_type == "predator" else self.num_agents
        self.local_agents_idxs = range(start_idx, end_idx)

        # Learning rate for actor-critic models
        self.critic_lr = 1e-4
        self.actor_lr = 5e-5

        # Creating Optimizer for Actor and Critic networks
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Creating critic models
        self.critic_model, self.target_critic = (
            Critic(num_states, num_actions, self.num_agents).model,
            Critic(num_states, num_actions, self.num_agents).model,
        )
        self.target_critic.set_weights(self.critic_model.get_weights())
        self.target_critic.trainable = False
        self.critic_model.compile(loss="mse", optimizer=self.critic_optimizer)

        # Creating target actor model
        self.actor_model, self.target_actor = (
            Actor(num_states).model,
            Actor(num_states).model,
        )
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_actor.trainable = False
        self.actor_model.compile(loss="mse", optimizer=self.actor_optimizer)
        self.actor_sender_controller = actor_sender_controller
        # Discount factor for future rewards
        self.gamma = 0.95
        # Used to update target networks
        self.tau = 0.005
        self.save_path = save_path
        # Send initial actors without training
        self.actor_sender_controller.send_actors([self.actor_model])

    def update(self):
        """
        Updates the Actor-Critic network of each agent, following the MADDPG algorithm.
        :return:
        """
        self.__update_actors(self.__update_critics())
        self.__update_targets()
        self.actor_sender_controller.send_actors([self.actor_model])

    @tf.function
    def __update_targets(self):
        """
        Slowly updates target parameters according to the tau rate <<1
        :return:
        """
        target_weights, weights = (
            self.target_actor.variables,
            self.actor_model.variables,
        )
        for a, b in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def __update_critics(self):
        """
        Updates the Critic networks by reshaping the sampled data.
        :return:
        """
        # Batch a sample from the buffer waiting one second between the requests
        # if there isn't any yet recorded
        logging.info("Sampling data from buffer")
        data_as_tuple = self.replay_buffer_controller.sample_batch()
        while data_as_tuple is None:
            sleep(1)
            logging.info("Sampling data from buffer")
            data_as_tuple = self.replay_buffer_controller.sample_batch()

        (state_batch, action_batch, reward_batch, next_state_batch) = data_as_tuple
        target_actions = []
        for j in range(self.num_agents):
            target_actions.append(
                self.target_actor(
                    # get the next state of the j-agent
                    next_state_batch[
                    :, j * self.num_states: (j + 1) * self.num_states
                    ],
                    training=True,
                )
            )
        action_batch_reshape = []
        for j in range(self.num_agents):
            action_batch_reshape.append(
                action_batch[:, j * self.num_actions: (j + 1) * self.num_actions]
            )
        return self.__update_critic_networks(
            state_batch,
            reward_batch,
            action_batch_reshape,
            next_state_batch,
            target_actions,
        )

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
        for i in self.local_agents_idxs:
            # Train the Critic network

            with tf.GradientTape() as tape:
                # Compute y = r_i + gamma * Q_i'(x',a_1',a_2', ...,a_n')
                y = reward_batch[:, i] + self.gamma * self.target_critic(
                    [next_state_batch, target_actions], training=True
                )
                # Eval Q_i(x, a_1, a_2, ..., a_n)
                critic_value = self.critic_model(
                    [state_batch, action_batch], training=True
                )
                # Compute loss = square_mean(y - Q_i(x, a_1, a_2, ..., a_n))
                # Here square_mean is taken over the batch
                critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

            critic_grad = tape.gradient(
                critic_loss, self.critic_model.trainable_variables
            )
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_model.trainable_variables)
            )
        return state_batch

    def __update_actors(self, state_batch):
        """
        Updates the Actor networks by:
        - Computing the loss from the Q-value of each agent Critic
        - Applying gradient to the Actor network
        :param state_batch: state batch
        :return:
        """
        actions = []
        for j in range(self.num_agents):
            actions.append(
                self.actor_model(
                    state_batch[:, j * self.num_states: (j + 1) * self.num_states],
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
        for i in self.local_agents_idxs:
            with tf.GradientTape(persistent=True) as tape:
                action = self.actor_model(
                    [state_batch[:, i * self.num_states: (i + 1) * self.num_states]],
                    training=True,
                )
                critic_value = self.critic_model(
                    [
                        state_batch,
                        [
                            [actions[k][:]] if k != i else action
                            for k in range(self.num_agents)
                        ],
                    ],
                    training=True,
                )
                actor_loss = -tf.math.reduce_mean(critic_value)

            actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_model.trainable_variables)
            )
