import jax
import jax.numpy as jnp

from ._jax import JAXRed as _JR


class JAXVectorLoop(_JR):
    def compute(self):
        zc = self.z.conj()

        for i, (a, b) in enumerate(self.pairs):
            self.out[i] = jnp.dot(self.z[a], zc[b])

        return jax.device_get(self.out)
