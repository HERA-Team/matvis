from ._cp import _CuPyRed, cp
import numpy as np
from ._rowswap import get_submatrices
from time import perf_counter

class CpRowSwap(_CuPyRed):
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
            cp.array(np.sort(a)) for a in self.antlist
        ]
        self.ant2list = [
            cp.array(np.sort(list(set(sum((self.antmap[a] for a in antlist), start=[]))))) for antlist in self.antlist
        ]

        self.out = [
            cp.zeros(shape=(len(a), len(b)), dtype=self._z.dtype) for a, b in zip(self.ant1list, self.ant2list)
        ]


        # TODO: really should figure out the computable size automatically.
        if max(self._z.shape) > 12*256**2:
            nsrc = self._z.shape[1]
            self.z = [slice(0, nsrc//4), slice(nsrc//4, 2*nsrc//4), slice(2*nsrc//4, 3*nsrc//4), slice(3*nsrc//4, None)]
        else:
            self.z = [cp.array(self._z)]

    def compute(self):
        outer = perf_counter()
        takes_time = 0
        gemms_time = 0
        mem_time = 0
        res = [0]*len(self.out)

        for _z in self.z:
            if isinstance(_z, slice):
                _z = cp.array(self._z[:, _z])

            nsrc = _z.shape[1]

            for a, b, out, r in zip(self.ant1list, self.ant2list, self.out, res):
                
                #start_gpu = cp.cuda.Event()
                #end_gpu = cp.cuda.Event()
                #start_gpu.record()
                za = _z[a]
                zb = _z[b]
                #end_gpu.record()
                #end_gpu.synchronize()
                #takes_time += cp.cuda.get_elapsed_time(start_gpu, end_gpu)

                #gemms_in = cp.cuda.Event()
                #gemms_out = cp.cuda.Event()
                #gemms_in.record()
                cp.dot(za, zb.conj().T, out=out)
                # self.gemm(
                #     self.h,
                #     "c",  # conjugate transpose for first (remember fortran order)
                #     "n",  # no transpose for second.
                #     len(a),
                #     len(b),
                #     nsrc,
                #     1.0,
                #     za.data,
                #     nsrc,
                #     zb.data,
                #     nsrc,
                #     0.0,
                #     out.data,
                #     len(a),
                # )
                #gemms_out.record()
                #gemms_out.synchronize()
                #gemms_time += cp.cuda.get_elapsed_time(gemms_in, gemms_out)


                t0 = perf_counter()
                r += out.get()
                mem_time += perf_counter() - t0
        outer -= perf_counter()

        print("Full compute time: ", -outer)
        print("take_along_axis compute time: ", takes_time)
        print("Gemms time: ", gemms_time)
        print("Memory return time: ", mem_time)
        return res
