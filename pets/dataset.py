import pickle
from typing import Tuple, Self
from dataclasses import dataclass

import numpy as np
from numpy.random import permutation

from jax import Array, numpy as jnp

Stats = Tuple[Array, Array]
Batch = Tuple[Array, Array, Array, Array]


class Dataset:
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
        self[idx] = (state, action, reward, next_state)
        self._steps = self._steps + 1

    def batches(self, batch_dim: int):
        indices = np.array([permutation(len(self)) for _ in range(self.ensemble_dim)])
        for idx in range(0, len(self), batch_dim):
            batch_indices = indices[:, idx : idx + batch_dim].T
            yield self[batch_indices]

    def stats(self) -> Stats:
        states, actions = self.states[: len(self)], self.actions[: len(self)]
        inputs = jnp.concatenate([states, actions], axis=-1)

        mean = jnp.mean(inputs, axis=0, keepdims=True)
        std = jnp.std(inputs, axis=0, keepdims=True)
        std = jnp.where(std < 1e-12, 1.0, std)
        return mean, std

    def __setitem__(self, idx, value):
        state, action, reward, next_state = value
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state

    def __getitem__(self, idx) -> Tuple[Array, Array, Array, Array]:
        return (
            jnp.array(self.states[idx]),
            jnp.array(self.actions[idx]),
            jnp.array(self.rewards[idx]),
            jnp.array(self.next_states[idx]),
        )

    def __len__(self):
        return min(self._steps, self.buffer_size)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> Self:
        with open(filename, "rb") as f:
            return pickle.load(f)
