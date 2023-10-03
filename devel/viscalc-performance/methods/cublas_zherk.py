from ._cublas import CuBLAS as _CB


class CuBLASZherk(_CB):
    def compute(self):
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
