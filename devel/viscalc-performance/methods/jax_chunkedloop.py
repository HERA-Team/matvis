from ._jax import JAXRed as _JR
import jax.numpy as jnp
import jax

class JAXChunkedLoop(_JR):
    chunksize = 350
            
    def compute(self):
        nchunks = len(self.ant1) // self.chunksize

        def doslc(slc):
            s1 = self.z[self.ant1[slc]]
            s2 = self.z[self.ant2[slc]]
            s1 *= s2.conj()
            sm = jnp.sum(s1, axis=1)
            self.out = self.out.at[slc].set(sm)

        for chunk in range(nchunks):
            slc = slice(chunk*self.chunksize, (chunk+1)*self.chunksize)
            doslc(slc)

        if len(self.ant1) % self.chunksize:
            slc = slice((chunk+1)*self.chunksize, None)
            doslc(slc)

        return jax.device_get(self.out)
