// Fused beam-interpolation kernels evaluating every (beam, feed, axis)
// combination for every source in a single launch. The original implementation
// made nbeam*nfeed*nax separate map_coordinates launches per chunk, which left
// the GPU idle most of the time waiting on the host to issue work.
//
// Grid: x indexes sources, y indexes (beam, axis, feed) combinations.
// Input layout  (nbeam, nax, nfeed, nza, naz) [UVBeam order],
// Output layout (nbeam, nfeed, nax, nsrc)     [matvis order].
// Out-of-range coordinates clamp to the grid edge instead of extrapolating.
//
// The `bicubic_*` kernels expect `beam` to hold cubic B-spline *coefficients*
// rather than raw grid values, with a CUBIC_HALO-node halo on each side of both
// grid axes (i.e. stored as nza+2 by naz+2). `nza`/`naz` are the underlying grid
// dimensions either way, so both orders take identical arguments. See
// matvis.gpu.beams.prefilter_beam, whose _CUBIC_HALO must match CUBIC_HALO here.
//
// Used by matvis.gpu.beams.gpu_beam_interpolation.
#include <cupy/complex.cuh>

// Coefficient nodes the cubic stencil reaches beyond each edge of the grid.
#define CUBIC_HALO 1

// blockIdx.y enumerates (beam, axis, feed) in the input array's own order, so
// it doubles as the input plane index. The output orders feeds before axes, so
// its plane index has to be rebuilt from the components; the beam index is
// reported too, since the grid spacings are per-beam.
__device__ inline long out_plane(const int p, const int nfeed, const int nax, int &bm)
{
    bm = p / (nax * nfeed);
    const int r = p % (nax * nfeed);
    return ((long)bm * nfeed + (r % nfeed)) * nax + (r / nfeed);
}

// Locate the grid cell containing `t` (in grid units): `i0` is the cell's lower
// node and `f` the fractional offset within it. Clamping `i0` to [0, n-2] both
// implements clamp-to-edge for out-of-range points and keeps the bicubic
// stencil (nodes i0-1 .. i0+2) within a CUBIC_HALO-node halo.
template<typename R>
__device__ inline void locate_cell(const R t, const long n, long &i0, R &f)
{
    i0 = (long)floor(t);
    f = t - i0;
    if (i0 < 0)     { i0 = 0;     f = 0; }
    if (i0 > n - 2) { i0 = n - 2; f = 1; }
}

// Cubic B-spline basis at the four nodes i0-1 .. i0+2 of the located cell.
template<typename R>
__device__ inline void cubic_weights(const R f, R w[4])
{
    const R f2 = f * f, f3 = f2 * f;
    const R sixth = (R)(1.0 / 6.0);
    w[0] = sixth * (1 - 3 * f + 3 * f2 - f3);
    w[1] = sixth * (4 - 6 * f2 + 3 * f3);
    w[2] = sixth * (1 + 3 * f + 3 * f2 - 3 * f3);
    w[3] = sixth * f3;
}

// ORDER is a template parameter so the unused branch below folds away at
// compile time; a plain `if` (rather than C++17's `if constexpr`) keeps this
// compiling under whatever standard NVRTC defaults to.
template<int ORDER, typename R, typename T>
__device__ void interp_all_planes(
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

    int bm;
    const long op = out_plane(blockIdx.y, nfeed, nax, bm);

    long x0, y0;
    R fx, fy;
    locate_cell(za[s] / dza[bm], nza, x0, fx);
    locate_cell((az[s] - azmin[bm]) / daz[bm], naz, y0, fy);

    T v;
    if (ORDER == 1) {
        const T* b = beam + ((long)blockIdx.y * nza + x0) * naz + y0;
        v = b[0]       * ((1 - fx) * (1 - fy))
          + b[1]       * ((1 - fx) * fy)
          + b[naz]     * (fx * (1 - fy))
          + b[naz + 1] * (fx * fy);
    } else {
        R wx[4], wy[4];
        cubic_weights(fx, wx);
        cubic_weights(fy, wy);

        // The halo shifts every coefficient index up by CUBIC_HALO, so grid node
        // (x0 - 1 + i, y0 - 1 + j) lives at coefficient (x0 + i, y0 + j).
        const long ncza = nza + 2 * CUBIC_HALO, ncaz = naz + 2 * CUBIC_HALO;
        const T* c = beam + ((long)blockIdx.y * ncza + x0) * ncaz + y0;

        v = T(0);
        for (int i = 0; i < 4; ++i) {
            T row = T(0);
            for (int j = 0; j < 4; ++j) row = row + c[i * ncaz + j] * wy[j];
            v = v + row * wx[i];
        }
    }

    out[op * nsrc + s] = v;
}

// One entry point per (order, dtype). `R` is the real type of the coordinates,
// `T` the (possibly complex) type of the beam.
#define BEAM_KERNEL(name, order, R, T)                                         \
extern "C" __global__ void name(                                               \
    const T* beam, const R* az, const R* za, const R* daz, const R* dza,       \
    const R* azmin, int nfeed, int nax, long nza, long naz, long nsrc, T* out)  \
{                                                                              \
    interp_all_planes<order, R, T>(                                            \
        beam, az, za, daz, dza, azmin, nfeed, nax, nza, naz, nsrc, out);       \
}

BEAM_KERNEL(bilinear_c64,  1, float,  complex<float>)
BEAM_KERNEL(bilinear_c128, 1, double, complex<double>)
BEAM_KERNEL(bilinear_f32,  1, float,  float)
BEAM_KERNEL(bilinear_f64,  1, double, double)

BEAM_KERNEL(bicubic_c64,   3, float,  complex<float>)
BEAM_KERNEL(bicubic_c128,  3, double, complex<double>)
BEAM_KERNEL(bicubic_f32,   3, float,  float)
BEAM_KERNEL(bicubic_f64,   3, double, double)
