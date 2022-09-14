import jax
import jax.numpy as jnp


def decode(x):
    x = jnp.max(x, 0)
    log_p_y = jax.nn.log_softmax(x, axis=1)
    return log_p_y
