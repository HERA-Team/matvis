from ._jax import JAXRed as _JR
import jax.numpy as jnp
import jax

class JAXVectorLoop(_JR):            
    def compute(self):
        zc = self.z.conj()

        for i, (a, b) in enumerate(self.pairs):
            self.out[i] = jnp.dot(self.z[a], zc[b])

        return jax.device_get(self.out)
