from typing import Callable

from jax import numpy as jnp, random as jr, tree_util as jtu, nn


def plan(
    state: jnp.ndarray,
    rollout_fn: Callable,
    action_dim: int,
    key: jr.PRNGKey,
    num_steps: int = 12,
    num_samples: int = 30,
    topk_ratio: float = 0.2,
    alpha: float = 0.2,
    iters: int = 5,
):
    topk = int(jnp.ceil(topk_ratio * num_samples))

    probs = jnp.full((num_steps, num_samples, action_dim), 1.0 / action_dim)
    state = jtu.tree_map(lambda x: jnp.repeat(x[None], num_samples, axis=0), state)
    for _ in range(iters):
        key, subkey = jr.split(key)
        actions = jr.categorical(subkey, jnp.log(probs + 1e-9))
        states, rewards = rollout_fn(state, actions[..., None])
        probs, r_top = _refit(probs, actions, rewards, topk, alpha)

    return probs, states, rewards


def _refit(probs, actions, rewards, topk, alpha):
    T, B, A = probs.shape

    cum_rewards = rewards.sum(0).squeeze(-1)[None]
    topk_indices = jnp.argsort(-cum_rewards, axis=-1)[..., :topk]
    a_top = jnp.take_along_axis(actions, topk_indices, axis=-1)
    r_top = jnp.take_along_axis(cum_rewards, topk_indices, axis=-1)

    one_hot_actions = nn.one_hot(a_top, A)
    counts = one_hot_actions.sum(axis=1)
    probs_new = counts / counts.sum(axis=-1, keepdims=True)

    probs_new = jnp.expand_dims(probs_new, axis=1).repeat(B, axis=1)
    probs_updated = alpha * probs_new + (1 - alpha) * probs
    return probs_updated, r_top
