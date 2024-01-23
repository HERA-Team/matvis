from ._cublas import _CuBLASRed
from . import _cublas as _cu
from time import perf_counter

class CuBLASVectorLoop(_CuBLASRed):
    def compute(self):
        total = perf_counter()
        dot = 0

        if self.transposed:
            size = self.z.shape[0]

            for i, (a, b) in enumerate(self.pairs):
                t0 = perf_counter()
                self.dotc(
                    self.h,
                    size,
                    self.z[:, b].gpudata,
                    1,
                    self.z[:, a].gpudata,
                    1,
                    self.out[i].gpudata,
                )
                _cu.sync()
                dot += perf_counter() - t0
        else:
            size = self.z.shape[1]
            t0 = perf_counter()
            for i, (a, b) in enumerate(self.pairs):
                self.dotc(
                    self.h,
                    size,
                    self.z[b].gpudata,
                    1,
                    self.z[a].gpudata,
                    1,
                    self.out[i].gpudata,
                )
            _cu.sync()
            dot += perf_counter() - t0

        get = perf_counter()
        res = self.out.get()
        get -= perf_counter()
        total -= perf_counter()

        print("Total time: ", -total)
        print("Dot time:", dot)
        print("Memory get time: ", get)

        return res
