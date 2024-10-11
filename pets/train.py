from typing import Tuple

import equinox as eqx
from jax import Array, numpy as jnp, vmap
from optax import GradientTransformation, OptState

from pets.model import Ensemble
from pets.dataset import Dataset, Normalizer

Batch = Tuple[Array, Array, Array, Array]


def train(
    model: Ensemble,
    optim: GradientTransformation,
    optim_state: OptState,
    dataset: Dataset,
    normalizer: Normalizer,
    batch_dim: int,
    num_epochs: int,
) -> tuple[Ensemble, OptState, list[float]]:
    normalizer = _update_normalizer(normalizer, dataset)

    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataset.batches(batch_dim)):
            model, optim_state, loss = train_step(model, batch, optim, optim_state, normalizer)
            total_loss = total_loss + loss.item()
        losses.append(total_loss / (batch_idx + 1))
    return model, optim_state, losses


@eqx.filter_jit
def train_step(
    model: Ensemble, batch: Batch, optim: GradientTransformation, optim_state: OptState, normalizer: Normalizer
) -> tuple[Ensemble, OptState, float]:
    loss, grads = loss_fn(model, batch, normalizer)
    updates, optim_state = optim.update(grads, optim_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss


@eqx.filter_value_and_grad
def loss_fn(model: Ensemble, batch: Batch, normalizer: Normalizer) -> float:
    state, action, reward, next_state = batch

    inputs = jnp.concatenate([state, action], axis=-1)
    inputs = normalizer.normalize(inputs)
    delta_mean, delta_logvar = vmap(model)(inputs)

    nll = _gaussian_nll(delta_mean, delta_logvar, next_state - state)
    return nll.mean((0, 2)).sum()


def _gaussian_nll(mean: Array, logvar: Array, target: Array) -> Array:
    return 0.5 * (jnp.exp(-logvar) * (mean - target) ** 2 + logvar)


def _update_normalizer(normalizer: Normalizer, dataset: Dataset) -> Normalizer:
    state, action, reward, next_state = dataset[: len(dataset)]
    inputs = jnp.concatenate([state, action], axis=-1)
    normalizer.update(inputs)
    return normalizer
