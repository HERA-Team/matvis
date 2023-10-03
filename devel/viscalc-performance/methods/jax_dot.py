from ._jax import JAXSolver as _JXS
import jax.numpy as jnp
import jax

class JAXDot(_JXS):
    def compute(self):
        out = jnp.dot(self.z, self.z.T.conj(), precision=jax.lax.Precision.HIGHEST)
        return jax.device_get(out)
    