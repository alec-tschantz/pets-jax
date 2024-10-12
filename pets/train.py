from typing import Tuple

import equinox as eqx
from jax import Array, numpy as jnp, vmap
from optax import GradientTransformation, OptState

from pets.model import Ensemble
from pets.dataset import Dataset, Stats, Batch


@eqx.filter_jit
def train_step(
    model: Ensemble, batch: Batch, stats: Stats, optim: GradientTransformation, optim_state: OptState
) -> tuple[Ensemble, OptState, float]:
    loss, grads = loss_fn(model, batch, stats)
    updates, optim_state = optim.update(grads, optim_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss


@eqx.filter_value_and_grad
def loss_fn(model: Ensemble, batch: Batch, stats: Stats) -> float:
    state, action, reward, next_state = batch
    delta_mean, delta_logvar = vmap(model, in_axes=(0, 0, None))(state, action, stats)
    nll = _gaussian_nll(delta_mean, delta_logvar, next_state - state)
    return nll.mean((0, 2)).sum()


def _gaussian_nll(mean: Array, logvar: Array, target: Array) -> Array:
    return 0.5 * (jnp.exp(-logvar) * (mean - target) ** 2 + logvar)
