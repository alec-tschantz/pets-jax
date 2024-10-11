import equinox as eqx
from jax import Array, numpy as jnp, vmap
from optax import GradientTransformation, OptState

from pets.models import Ensemble
from pets.training import Buffer

_to_jax = lambda batch: tuple([jnp.array(x) for x in batch])


def train(
    model: Ensemble,
    optim: GradientTransformation,
    optim_state: OptState,
    buffer: Buffer,
    batch_dim: int,
    num_epochs: int,
) -> tuple[Ensemble, OptState, list[float]]:
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(buffer.batches(batch_dim)):
            model, optim_state, loss = train_step(model, _to_jax(batch), optim, optim_state)
            total_loss = total_loss + loss.item()
        losses.append(total_loss / batch_idx)
    return model, optim_state, losses


@eqx.filter_jit
def train_step(
    model: Ensemble, batch: tuple[Array, Array, Array, Array], optim: GradientTransformation, optim_state: OptState
) -> tuple[Ensemble, OptState, float]:
    loss, grads = loss_fn(model, batch)
    updates, optim_state = optim.update(grads, optim_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss


@eqx.filter_value_and_grad
def loss_fn(model: Ensemble, batch: tuple[Array, Array, Array, Array]) -> float:
    state, action, reward, next_state = batch
    inputs = jnp.concatenate([state, action], axis=-1)
    delta_mean, delta_logvar = vmap(model)(inputs)

    delta_target = next_state - state
    nll = _gaussian_nll(delta_mean, delta_logvar, delta_target)
    return nll.mean((0, 2)).sum()


def _gaussian_nll(mean: Array, logvar: Array, target: Array) -> Array:
    return 0.5 * (jnp.exp(-logvar) * (mean - target) ** 2 + logvar)
