from ._cublas import _CuBLAS as _CB


class CuBLASZherk(_CB):
    def compute(self):
        if self.transposed:
            nsrc, nant = self.z.shape
            self.herk(
                self.h,
                uplo="L",
                trans="n",
                n=nant,
                k=nsrc,
                alpha=1.0,
                A=self.z.gpudata,
                lda=nant,
                beta=0.0,
                C=self.out.gpudata,
                ldc=nant,
            )
        else:
            nant, nsrc = self.z.shape

            self.herk(
                self.h,
                uplo="L",
                trans="c",
                n=nant,
                k=nsrc,
                alpha=1.0,
                A=self.z.gpudata,
                lda=nsrc,
                beta=0.0,
                C=self.out.gpudata,
                ldc=nant,
            )
        return self.out.get()
