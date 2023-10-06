from ._cublas import _CuBLASRed

class CuBLASVectorLoop(_CuBLASRed):
    def compute(self):
        if self.transposed:
            size = self.z.shape[0]

            for i, (a, b) in enumerate(self.pairs):
                self.dotc(
                    self.h, size, self.z[:, b].gpudata, 1, self.z[:, a].gpudata, 1, self.out[i].gpudata
                )
        else:
            size = self.z.shape[1]

            for i, (a, b) in enumerate(self.pairs):
                self.dotc(
                    self.h, size, self.z[b].gpudata, 1, self.z[a].gpudata, 1, self.out[i].gpudata
                )

        return self.out.get()
