from ._lib import RedundantSolver, Solver

try:
    import jax
    import jax.numpy as jnp
    from jax import config

    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False


class JAXSolver(Solver):
    def setup(self):
        if self._z.dtype.name == "complex128":
            config.update("jax_enable_x64", True)

        self.z = jax.device_put(self._z)


class JAXRed(RedundantSolver):
    def setup(self):
        if self._z.dtype.name == "complex128":
            config.update("jax_enable_x64", True)

        self.z = jax.device_put(self._z)
        self.ant1 = jax.device_put(self.pairs[:, 0])
        self.ant2 = jax.device_put(self.pairs[:, 1])
        self.out = jnp.empty(len(self.ant1), dtype=self._z.dtype)

    def compute(self):
        nchunks = len(self.ant1) // self.chunksize

        def doslc(slc):
            s1 = self.z[self.ant1[slc]]
            s2 = self.z[self.ant2[slc]]
            s1 *= s2.conj()
            sm = jnp.sum(s1, axis=1)
            self.out = self.out.at[slc].set(sm)

        for chunk in range(nchunks):
            slc = slice(chunk * self.chunksize, (chunk + 1) * self.chunksize)
            doslc(slc)

        if len(self.ant1) % self.chunksize:
            slc = slice((chunk + 1) * self.chunksize, None)
            doslc(slc)

        return jax.device_get(self.out)
