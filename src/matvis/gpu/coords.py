"""Coordinate rotation methods for the GPU."""

import cupy as cp

from ..cpu.coords import CoordinateRotationERFA

_ld_kernel_code = r"""
const double ERFA_SRS=1.97412574336e-8;

extern "C" __global__ void light_deflection(REAL_DTYPE* p, int np, const double* e, const double em, const double dlim) {
    double qpqde, w;

    int k0 = (blockDim.x * blockIdx.x + threadIdx.x);
    int k1 = k0+np;
    int k2 = k1+np;

    qpqde = p[k0]*(p[k0]+e[0]) + p[k1]*(p[k1]+e[1]) + p[k2]*(p[k2]+e[2]);
    if(qpqde<dlim){
        qpqde = dlim;
    }
    w = ERFA_SRS / em / qpqde;

    p[k0] += w*(-e[1]*p[k0]*p[k1] + e[0]*p[k1]*p[k1] - e[2]*p[k0]*p[k2] + e[0]*p[k2]*p[k2]);
    p[k1] += w*(+e[1]*p[k0]*p[k0] - e[0]*p[k0]*p[k1] - e[2]*p[k1]*p[k2] + e[1]*p[k2]*p[k2]);
    p[k2] += w*(+e[2]*p[k0]*p[k0] + e[2]*p[k1]*p[k1] - e[0]*p[k0]*p[k2] - e[1]*p[k1]*p[k2]);
}
"""

ld_kernel_single = cp.RawKernel(
    _ld_kernel_code.replace("REAL_DTYPE", "float"), "light_deflection"
)
ld_kernel_double = cp.RawKernel(
    _ld_kernel_code.replace("REAL_DTYPE", "double"), "light_deflection"
)


class GPUCoordinateRotationERFA(CoordinateRotationERFA):
    """Perform coordinate rotation with functions pulled from ERFA.

    This is a subclass of :class:`matvis.cpu.coords.CoordinateRotationERFA` that
    over-rides the light-deflection function to be computed as a custom CUDA kernel.

    All other methods in the super-class are compatible both with GPU and CPU
    intrinsically.
    """

    requires_gpu: bool = True

    def _ld(self, p, e, em, dlim):
        if self.precision == 1:
            ld_kernel_single((p.shape[1],), (1,), (p, p.shape[1], e, em, dlim))
        else:
            ld_kernel_double((p.shape[1],), (1,), (p, p.shape[1], e, em, dlim))
