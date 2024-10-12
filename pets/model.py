from typing import Callable, Optional, Tuple

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn, vmap, lax

from pets.dataset import Stats


def _normalize(stats: Stats, inputs: Array) -> Array:
    mean, std = stats
    return (inputs - mean) / std


class EnsembleLinear(eqx.Module):
    weights: Array
    biases: Array

    def __init__(self, input_dim: int, output_dim: int, ensemble_dim: int, *, key: jr.PRNGKey):
        std = 1 / (2 * jnp.sqrt(input_dim))

        self.weights = std * jr.truncated_normal(key, -2, 2, shape=(ensemble_dim, input_dim, output_dim))
        self.biases = jnp.zeros((ensemble_dim, output_dim))

    def __call__(self, x: Array) -> Array:
        return jnp.einsum("ij,ijk->ik", x, self.weights) + self.biases


class Ensemble(eqx.Module):
    fc_1: EnsembleLinear
    fc_2: EnsembleLinear
    fc_3: EnsembleLinear
    act_fn: Callable[[Array], Array]
    min_logvar: float
    max_logvar: float
    ensemble_dim: int

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        ensemble_dim: int,
        act_fn: Callable[Array, Array] = nn.leaky_relu,
        *,
        key: jr.PRNGKey,
    ):
        keys = jr.split(key, 4)
        self.fc_1 = EnsembleLinear(input_dim, hidden_dim, ensemble_dim, key=keys[1])
        self.fc_2 = EnsembleLinear(hidden_dim, hidden_dim, ensemble_dim, key=keys[2])
        self.fc_3 = EnsembleLinear(hidden_dim, output_dim * 2, ensemble_dim, key=keys[3])

        self.act_fn = act_fn
        self.ensemble_dim = ensemble_dim

        self.max_logvar = -1.0
        self.min_logvar = -5.0

    def __call__(self, state: Array, action: Array, stats: Stats) -> Tuple[Array]:
        x = jnp.concatenate([state, action], axis=-1)
        x = _normalize(stats, x)

        x = self.fc_1(x)
        x = self.act_fn(x)
        x = self.fc_2(x)
        x = self.act_fn(x)
        x = self.fc_3(x)

        delta_mean, delta_logvar = jnp.split(x, 2, axis=-1)
        delta_logvar = self.max_logvar - nn.softplus(self.max_logvar - delta_logvar)
        delta_logvar = self.min_logvar + nn.softplus(delta_logvar - self.min_logvar)

        return delta_mean, delta_logvar

    @eqx.filter_jit
    def rollout(self, state, actions, stats, key):
        _tile = lambda x: jnp.tile(x[:, None, ...], (1, self.ensemble_dim, 1))
        state, actions = _tile(state), vmap(_tile)(actions)

        def scan_fn(carry, action):
            state, key = carry
            key, subkey = jr.split(key)

            delta_mean, delta_logvar = vmap(self, in_axes=(0, 0, None))(state, action, stats)
            delta_std = jnp.sqrt(jnp.exp(delta_logvar))

            delta = delta_mean + delta_std * jr.normal(key, delta_mean.shape)
            next_state = state + delta
            return (next_state, key), next_state

        (final_state, _), states = lax.scan(scan_fn, (state, key), actions)
        return states
