// Mirror the (valid) lower triangle of a column-major hermitian matrix into
// the upper triangle. Used by matvis.gpu._cublas._mirror_hermitian after a
// cuBLAS herk call, which only fills one triangle.
#include <cupy/complex.cuh>
extern "C" {
__global__ void mirror_c(complex<float>* C, long n) {
    long p = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (p >= n * n) return;
    long c = p / n, r = p % n;  // column-major: p = c*n + r
    if (r < c) C[p] = conj(C[r * n + c]);
}
__global__ void mirror_z(complex<double>* C, long n) {
    long p = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (p >= n * n) return;
    long c = p / n, r = p % n;
    if (r < c) C[p] = conj(C[r * n + c]);
}
}
