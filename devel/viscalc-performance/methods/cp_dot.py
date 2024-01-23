from ._cp import _CuPy
import cupy as cp

class CpDot(_CuPy):
    def compute(self):
        if self.transposed:
            cp.dot(self.z.T, self.z.conj(), out=self.out)
        else:
            cp.dot(self.z, self.z.conj().T, out=self.out)
        return self.out.get()