// One fused bilinear-interpolation kernel evaluating every (beam, feed, axis)
// combination for every source in a single launch. The previous implementation made
// nbeam*nfeed*nax separate map_coordinates launches per chunk (1400 launches
// for a 350-antenna array with per-antenna beams), which left the GPU idle
// most of the time waiting on the host to issue work.
//
// Grid: x indexes sources, y indexes (beam, feed, axis) combinations.
// Out-of-range coordinates clamp to the grid edge. Input layout
// (nbeam, nax, nfeed, nza, naz) [UVBeam order], output layout
// (nbeam, nfeed, nax, nsrc) [matvis order].
// Used by matvis.gpu.beams.gpu_beam_interpolation.
#include <cupy/complex.cuh>

template<typename R, typename T>
__device__ void bilinear_all_planes(
    const T* __restrict__ beam,
    const R* __restrict__ az,
    const R* __restrict__ za,
    const R* __restrict__ daz,
    const R* __restrict__ dza,
    const R* __restrict__ azmin,
    const int nfeed,
    const int nax,
    const long nza,
    const long naz,
    const long nsrc,
    T* __restrict__ out)
{
    const long s = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (s >= nsrc) return;
    const int p = blockIdx.y;
    const int bm = p / (nax * nfeed);
    const int r = p % (nax * nfeed);
    const int ax = r / nfeed;
    const int fd = r % nfeed;

    R x = za[s] / dza[bm];
    R y = (az[s] - azmin[bm]) / daz[bm];

    long x0 = (long)floor(x);
    long y0 = (long)floor(y);
    R fx = x - x0;
    R fy = y - y0;
    // Out-of-range points clamp to the grid edge instead of extrapolating.
    if (x0 < 0) { x0 = 0; fx = 0; }
    if (x0 > nza - 2) { x0 = nza - 2; fx = 1; }
    if (y0 < 0) { y0 = 0; fy = 0; }
    if (y0 > naz - 2) { y0 = naz - 2; fy = 1; }

    const T* b = beam + ((((long)bm * nax + ax) * nfeed + fd) * nza + x0) * naz + y0;
    T v = b[0] * ((1 - fx) * (1 - fy))
        + b[1] * ((1 - fx) * fy)
        + b[naz] * (fx * (1 - fy))
        + b[naz + 1] * (fx * fy);

    out[(((long)bm * nfeed + fd) * nax + ax) * nsrc + s] = v;
}

extern "C" {
__global__ void bilinear_c64(
    const complex<float>* beam, const float* az, const float* za,
    const float* daz, const float* dza, const float* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, complex<float>* out)
{ bilinear_all_planes<float, complex<float> >(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }

__global__ void bilinear_c128(
    const complex<double>* beam, const double* az, const double* za,
    const double* daz, const double* dza, const double* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, complex<double>* out)
{ bilinear_all_planes<double, complex<double> >(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }

__global__ void bilinear_f32(
    const float* beam, const float* az, const float* za,
    const float* daz, const float* dza, const float* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, float* out)
{ bilinear_all_planes<float, float>(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }

__global__ void bilinear_f64(
    const double* beam, const double* az, const double* za,
    const double* daz, const double* dza, const double* azmin,
    int nfeed, int nax, long nza, long naz, long nsrc, double* out)
{ bilinear_all_planes<double, double>(beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out); }
}
