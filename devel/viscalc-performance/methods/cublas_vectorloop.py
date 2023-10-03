from ._cublas import _CuBLASRed

class CuBLASVectorLoop(_CuBLASRed):
    def compute(self):
        size = self.z.shape[1]

        for i, (a, b) in enumerate(self.pairs):
            self.out[i] = self.dotc(self.h, size, self.z[b].gpudata, 1, self.z[a].gpudata, 1)

        return self.out.get()