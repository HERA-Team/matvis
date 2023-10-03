from ._cublas import CuBLAS as _CB

class CuBLASZgemm(_CB):
    def compute(self):
        nant, nsrc = self.z.shape

        self.gemm(
            self.h,
            "c",  # conjugate transpose for first (remember fortran order)
            "n",  # no transpose for second.
            nant,
            nant,
            nsrc,
            1.0,
            self.z.gpudata,
            nsrc,
            self.z.gpudata,
            nsrc,
            0.0,
            self.out.gpudata,
            nant,
        )
        return self.out.get()