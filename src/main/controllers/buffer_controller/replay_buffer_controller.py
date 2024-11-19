class ReplayBufferController:
    def sample_batch(self) -> tuple:
        """
        Sample a data batch of batch_size
        :return: (state_batch, action_batch, reward_batch, next_state_batch) tuple
        """
        raise NotImplementedError("Subclasses must implement this method")
