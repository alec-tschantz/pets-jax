import equinox as eqx
from jax import Array, numpy as jnp, random as jr


class EnsembleLinear(eqx.Module):
    weights: Array
    biases: Array

    def __init__(self, input_dim: int, output_dim: int, ensemble_dim: int, *, key: jr.PRNGKey):
        std = 1 / (2 * jnp.sqrt(input_dim))

        self.weights = std * jr.truncated_normal(key, -2, 2, shape=(ensemble_dim, input_dim, output_dim))
        self.biases = jnp.zeros((ensemble_dim, output_dim))

    def __call__(self, x: Array) -> Array:
        return jnp.einsum("ij,ijk->ik", x, self.weights) + self.biases
