"""Some small library code to help with implementing different solvers."""

import numpy as np


class Solver:
    def __init__(self, z: np.ndarray):
        self._z = z
        
    def setup(self):
        self.z = self._z
    
    def compute(self) -> np.ndarray:
        pass
    
    def teardown(self):
        pass

    def __call__(self) -> np.ndarray:
        self.setup()
        out = self.compute()
        self.teardown()
        return out
    
    @classmethod
    def test(cls, z0: np.ndarray, v0: np.ndarray, rtol=None, atol=None):
        if rtol is None:
            rtol = 1e-5 if z0.dtype is complex else 1e-3
        if atol is None:
            atol = 1e-5 if z0.dtype is complex else 1e-3

        obj = cls(z0)
        result = obj()
        np.testing.assert_allclose(np.triu(result), np.triu(v0), rtol=rtol, atol=atol)
        
class RedundantSolver(Solver):
    def __init__(self, z: np.ndarray, pairs: np.ndarray):
        self._z = z
        self.pairs = pairs
            
    @classmethod
    def test(cls, z0: np.ndarray, v0: np.ndarray, rtol=None, atol=None):
        if rtol is None:
            rtol = 1e-5 if z0.dtype is complex else 1e-3
        if atol is None:
            atol = 1e-5 if z0.dtype is complex else 1e-3

        # All the pairs.
        nant = z0.shape[0]
        pairs = np.array([(a, b) for a in range(nant) for b in range(nant)])
        obj = cls(z0, pairs)
        result = obj()
        np.testing.assert_allclose(np.triu(result.reshape((nant, nant))), np.triu(v0), rtol=rtol, atol=atol)
    