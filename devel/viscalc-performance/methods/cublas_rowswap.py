from ._cublas import _CuBLASRed, cuda_take_along_axis
from . import _cublas as _cu
import numpy as np
from ._rowswap import get_submatrices
from time import perf_counter

class CuBLASRowSwap(_CuBLASRed):
    """This follows a simple method to try make the matrix as dense as possible.
    
    For now, it uses only the default pairs given (it does not attempt to try use
    different pairs in the redundant sets). It orders the ants in order of how many 
    unique pairs they appear in, both for the first ant and second ant (rows and cols).
    It then splits the matrix into some number of sub-matrices that attempt to maximise
    the density in each.
    """
    def __init__(self, *args, revpairs: bool = False, nsubs: int=3, **kwargs ):
        super().__init__(*args, **kwargs)
        self.revpairs = revpairs
        self.nsubs = nsubs

    def setup(self):
        self.create()

        self.antmap, antlists = get_submatrices(
            self.pairs[:, ::-1] if self.revpairs else self.pairs, 
            max_subs=self.nsubs
        )
        self.antlist = antlists[-1]

        self.ant1list = [
            _cu.gpuarray.to_gpu(np.sort(a)) for a in self.antlist
        ]
        self.ant2list = [
            _cu.gpuarray.to_gpu(np.sort(list(set(sum((self.antmap[a] for a in antlist), start=[]))))) for antlist in self.antlist
        ]

        self.out = [
            _cu.gpuarray.zeros(shape=(len(a), len(b)), dtype=self._z.dtype) for a, b in zip(self.ant1list, self.ant2list)
        ]


        # TODO: really should figure out the computable size automatically.
        if max(self._z.shape) > 12*256**2:
            nsrc = self._z.shape[1]
            self.z = [slice(0, nsrc//4), slice(nsrc//4, 2*nsrc//4), slice(2*nsrc//4, 3*nsrc//4), slice(3*nsrc//4, None)]
        else:
            self.z = [_cu.gpuarray.to_gpu(self._z)]

    def compute(self):
        outer = perf_counter()
        takes_time = 0
        gemms_time = 0
        mem_time = 0
        res = [0]*len(self.out)

        for _z in self.z:
            if isinstance(_z, slice):
                _z = _cu.gpuarray.to_gpu(self._z[:, _z])

            nsrc = _z.shape[1]

            for a, b, out, r in zip(self.ant1list, self.ant2list, self.out, res):
                takes_in = perf_counter()
                za = cuda_take_along_axis(_z, a)
                zb = cuda_take_along_axis(_z, b)
                _cu.autoinit.context.synchronize()
                takes_time += perf_counter() - takes_in

                gemms_in = perf_counter()
                self.gemm(
                    self.h,
                    "c",  # conjugate transpose for first (remember fortran order)
                    "n",  # no transpose for second.
                    len(a),
                    len(b),
                    nsrc,
                    1.0,
                    za.gpudata,
                    nsrc,
                    zb.gpudata,
                    nsrc,
                    0.0,
                    out.gpudata,
                    len(a),
                )
                _cu.autoinit.context.synchronize()
                gemms_time += perf_counter() - gemms_in


                t0 = perf_counter()
                r += out.get()
                mem_time += perf_counter() - t0
        outer -= perf_counter()

        print("Full compute time: ", -outer)
        print("take_along_axis compute time: ", takes_time)
        print("Gemms time: ", gemms_time)
        print("Memory return time: ", mem_time)
        return res
