from typing import Callable

import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn

from pets.models import EnsembleLinear


class Ensemble(eqx.Module):
    fc_1: EnsembleLinear
    fc_2: EnsembleLinear
    fc_3: EnsembleLinear
    act_fn: Callable[[Array], Array]
    min_logvar: float
    max_logvar: float

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
        keys = jr.split(key, 3)
        self.fc_1 = EnsembleLinear(input_dim, hidden_dim, ensemble_dim, key=keys[0])
        self.fc_2 = EnsembleLinear(hidden_dim, hidden_dim, ensemble_dim, key=keys[1])
        self.fc_3 = EnsembleLinear(hidden_dim, output_dim * 2, ensemble_dim, key=keys[2])

        self.act_fn = act_fn
        self.max_logvar = 0.5
        self.min_logvar = -10.0

    def __call__(self, x: Array) -> Array:
        x = self.fc_1(x)
        x = self.act_fn(x)
        x = self.fc_2(x)
        x = self.act_fn(x)
        x = self.fc_3(x)

        delta_mean, delta_logvar = jnp.split(x, 2, axis=-1)
        delta_logvar = self.max_logvar - nn.softplus(self.max_logvar - delta_logvar)
        delta_logvar = self.min_logvar + nn.softplus(delta_logvar - self.min_logvar)

        return delta_mean, delta_logvar
