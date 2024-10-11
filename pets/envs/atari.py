from typing import Tuple
from dataclasses import dataclass

import random
import numpy as np
from jax import numpy as jnp
import gymnasium as gym


ram_dict = {
    "player_y": 51,
    "enemy_y": 50,
    "ball_x": 49,
    "ball_y": 54,
}


def reward_fn(next_obs: jnp.array) -> jnp.array:
    ball_x, ball_y = next_obs[..., 2], next_obs[..., 3]
    mask = (ball_x == 205.0) & (ball_y == 0.0)
    return jnp.where(mask, -1, 0)


@dataclass
class Space:
    shape: Tuple[int]

    def sample(self):
        return np.array([random.randint(0, self.shape[0] - 1)])


class AtariEnv:
    def __init__(self, env_name: str, render_mode: str = "rgb_array"):
        self.env = gym.make(env_name, render_mode=render_mode)

        self.action_space = Space((3,))
        self.observation_space = Space((4 * 2,))
        self._idx_to_action = {0: 0, 1: 2, 2: 3}
        self._prev_state = np.zeros(4)

    def step(self, action: int):
        action = self._idx_to_action[action.item()]
        observation, reward, done, truncated, info = self.env.step(action)
        return self._state(), reward, done, truncated, info

    def reset(self):
        _ = self.env.reset()
        return self._state(), {}

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def _state(self) -> np.ndarray:
        ram = self.env.unwrapped.ale.getRAM()
        state = [ram[v] for v in ram_dict.values()]
        state = np.array(state, dtype=np.float32)
        delta_state = state - self._prev_state
        self._prev_state = state
        return np.concatenate([state, delta_state])
