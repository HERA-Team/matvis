import jax
import jax.numpy as jnp

from ._jax import JAXRed as _JR

@jax.jit
def _go_at_it(z: jnp.array, out: jnp.array, pairs: jnp.array):
    zc = z.conj()

    for i, (a, b) in enumerate(pairs):
        out = out.at[i].set(jnp.dot(z[a], zc[b]))

    return jax.device_get(out)


class JAXVectorLoop(_JR):
    def compute(self):
        return _go_at_it(self.z, self.out, self.pairs)
        