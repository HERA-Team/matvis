// Z[ant, feed, ax, src] = A[beam_idx[ant], feed, ax, src] * exptau[ant, src] * sqrtI[src]
//
// One elementwise pass replaces the previous implementation's nfeed*nax
// broadcast copies, the Python loop over antennas, and the full-device
// synchronize -- about 9 passes over the Z-sized array in total, plus host
// stalls, become a single pass. Used by matvis.gpu.getz.GPUZMatrixCalc.
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
