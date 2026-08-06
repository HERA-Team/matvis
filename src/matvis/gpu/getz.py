"""Fused GPU computation of the Z matrix."""

import cupy as cp
import numpy as np

from ..core.getz import ZMatrixCalc

# Z[ant, feed, ax, src] = A[beam_idx[ant], feed, ax, src] * exptau[ant, src] * sqrtI[src]
#
# One elementwise pass replaces the previous implementation's nfeed*nax
# broadcast copies, the Python loop over antennas, and the full-device
# synchronize — about 9 passes over the Z-sized array in total, plus host
# stalls, become a single pass.
_FUSED_Z_MODULE = cp.RawModule(
    code=r"""
#include <cupy/complex.cuh>

template<typename R, typename T>
__device__ void fused_z(
    const T* __restrict__ beam,      // (nbeam, nfeed, nax, nsrc)
    const T* __restrict__ exptau,    // (nant, nsrc)
    const R* __restrict__ sqrt_flux, // (nsrc,)
    const long* __restrict__ beam_idx,  // (nant,) or NULL
    const long bmul,                 // if beam_idx NULL: 0 -> one shared beam, 1 -> beam per ant
    const int nfeed,
    const int nax,
    const long nsrc,
    const long ntot,
    T* __restrict__ out)             // (nant, nfeed, nax, nsrc)
{
    const long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= ntot) return;
    const long s = i % nsrc;
    long rest = i / nsrc;
    const int ax = rest % nax;
    rest /= nax;
    const int fd = rest % nfeed;
    const long ant = rest / nfeed;
    const long bm = beam_idx == NULL ? ant * bmul : beam_idx[ant];

    const T a = beam[((bm * nfeed + fd) * nax + ax) * nsrc + s];
    out[i] = a * exptau[ant * nsrc + s] * sqrt_flux[s];
}

extern "C" {
__global__ void fused_z_c64(
    const complex<float>* beam, const complex<float>* exptau,
    const float* sqrt_flux, const long* beam_idx, long bmul,
    int nfeed, int nax, long nsrc, long ntot, complex<float>* out)
{ fused_z<float, complex<float> >(beam, exptau, sqrt_flux, beam_idx, bmul, nfeed, nax, nsrc, ntot, out); }

__global__ void fused_z_c128(
    const complex<double>* beam, const complex<double>* exptau,
    const double* sqrt_flux, const long* beam_idx, long bmul,
    int nfeed, int nax, long nsrc, long ntot, complex<double>* out)
{ fused_z<double, complex<double> >(beam, exptau, sqrt_flux, beam_idx, bmul, nfeed, nax, nsrc, ntot, out); }
}
"""
)


class GPUZMatrixCalc(ZMatrixCalc):
    """Compute the Z matrix on the GPU in a single fused kernel."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("gpu", True)
        super().__init__(*args, **kwargs)
        self._beam_idx_gpu = None

    def __call__(
        self,
        sqrt_flux: cp.ndarray,
        beam: cp.ndarray,
        exptau: cp.ndarray,
        beam_idx: np.ndarray | None,
    ) -> cp.ndarray:
        """Compute Z = A * sqrtI * exp(tau) in one pass.

        See :meth:`matvis.core.getz.ZMatrixCalc.__call__` for parameters.
        Unlike the base implementation, ``exptau`` is not modified in place.
        """
        if beam_idx is None:
            bidx = np.uint64(0)  # NULL pointer
            # A single beam is shared by all antennas; otherwise one per ant.
            bmul = np.int64(0 if beam.shape[0] == 1 else 1)
        else:
            if self._beam_idx_gpu is None:
                self._beam_idx_gpu = cp.asarray(beam_idx, dtype=np.int64)
            bidx = self._beam_idx_gpu
            bmul = np.int64(1)  # unused when beam_idx is given

        kern = _FUSED_Z_MODULE.get_function(
            "fused_z_c64" if self.ctype == np.complex64 else "fused_z_c128"
        )

        ntot = self.nant * self.nfeed * self.nax * self.nsrc
        block = 256
        rdtype = np.float32 if self.ctype == np.complex64 else np.float64
        sqrt_flux = cp.ascontiguousarray(sqrt_flux, dtype=rdtype)
        assert beam._c_contiguous and exptau._c_contiguous

        kern(
            ((ntot + block - 1) // block,),
            (block,),
            (
                beam,
                exptau,
                sqrt_flux,
                bidx,
                bmul,
                np.int32(self.nfeed),
                np.int32(self.nax),
                np.int64(self.nsrc),
                np.int64(ntot),
                self.z,
            ),
        )

        return self.z.reshape(self.nant * self.nfeed, self.nax * self.nsrc)
