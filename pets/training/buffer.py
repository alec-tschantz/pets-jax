from typing import Tuple
from dataclasses import dataclass

import numpy as np
from numpy.random import permutation


class Buffer:
    def __init__(self, state_dim: int, action_dim: int, ensemble_dim: int, buffer_size: int = 10**6):
        self.ensemble_dim = ensemble_dim
        self.buffer_size = buffer_size

        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.next_states = np.zeros((buffer_size, state_dim))

        self._steps = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray):
        idx = self._steps % self.buffer_size
        self._steps = self._steps + 1

        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state

    def batches(self, batch_dim: int):
        indices = np.array([permutation(len(self)) for _ in range(self.ensemble_dim)])

        for idx in range(0, len(self), batch_dim):
            batch_indices = indices[:, idx : idx + batch_dim].T
            yield (
                self.states[batch_indices],
                self.actions[batch_indices],
                self.rewards[batch_indices],
                self.next_states[batch_indices],
            )

    def __len__(self):
        return min(self._steps, self.buffer_size)
