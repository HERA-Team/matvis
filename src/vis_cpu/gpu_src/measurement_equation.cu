// CUDA code for computing "voltage" visibilities
// [A^1/2 * I^1/2 * exp(-2*pi*i*freq*dot(a,s)/c)]

// ========== Template Parameters =========
// This code is actually a Jinja template, which can use the following parameters. They
// appear in double curly braces, eg. {{ DTYPE }}. They get substituted before compilation.
//
// DTYPE    : float or double
// CDTYPE   : cuFloatComplex or cuDoubleComplex
// BLOCK_PX : # of sky pixels handled by one GPU block, used to size shared memory
// NANT     : # of antennas to pair into visibilities
// NPIX     : # of sky pixels to sum over.
// NAX      : # of E-field axes in the beam
// NFEED    : # of feeds in the beam
// -------------------------------------------------------------------------------------

#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

// Shared memory for storing per-antenna results to be reused among all ants
// for "BLOCK_PX" pixels, avoiding a rush on global memory.
__shared__ {{ DTYPE }} sh_buf[{{ BLOCK_PX }}*5];

// Compute A*I*exp(ij*tau*freq) for all antennas, storing output in v
__global__ void MeasEq(
    {{ CDTYPE }} *A, {{ DTYPE }} *sqrtI, {{ DTYPE }} *tau,
    {{ DTYPE }} freq, uint nsrc, uint *beam_idx, {{ CDTYPE }} *v
){
    const uint nbeam = {{ NBEAM }};
    const uint nax  = {{ NAX }};
    const uint nfeed= {{ NFEED }};
    const uint nant = {{ NANT }};

    const uint tx = threadIdx.x; // First dim is src
    const uint ty = threadIdx.y; // Second dim is ant

    const uint src  =  blockIdx.x * blockDim.x + threadIdx.x;  // first thread dim is src on sky
    const uint antax = blockIdx.y * blockDim.y + threadIdx.y;    // second thread dim is nax*nant
    const uint feed  = blockIdx.z * blockDim.z + threadIdx.z;   // third thread dim is nfeed

    uint ant = antax / nax;
    uint ax = antax % nax;
    uint beam = beam_idx[ant];

    {{ CDTYPE }} amp;
    {{ DTYPE }} phs;
    if (ant >= nant || ax >= nax || feed >= nfeed || src >= nsrc) return;
    if (ty == 0)
        sh_buf[tx] = sqrtI[src];
    __syncthreads(); // make sure all memory is loaded before computing

    // Create both real/imag parts of the "amplitude"
    uint tau_indx = ant*nsrc + src;
    uint Aindx = ax*(nfeed*nbeam*nsrc) + feed*nbeam*nsrc + beam*nsrc + src;
    amp.x = A[Aindx].x * sh_buf[tx];
    amp.y = A[Aindx].y * sh_buf[tx];

    phs = tau[tau_indx] * freq;

    uint vidx = feed*(nant*nax*nsrc) + ant*nax*nsrc + ax*nsrc + src;
    v[vidx] = cuCmul{{ f }}(amp, make_{{ CDTYPE }}(cos(phs), sin(phs)));

    __syncthreads(); // make sure everyone used mem before kicking out
}

__global__ void VisInnerProduct({{ CDTYPE }} *v, uint npix, {{ CDTYPE }} *vis)
{
    const uint nant = {{ NANT }};
    const uint nax  = {{ NAX }};
    const uint nfeed= {{ NFEED }};

    const uint row = blockIdx.y * blockDim.y + threadIdx.y; // nfeed * nant
    const uint col = blockIdx.x * blockDim.x + threadIdx.x; // nfeed * nant

    // v is the antenna-based "sqrt" visibility. It has shape (nax, nfeed, nant, npix).
    // To get the visibility, which has shape (nfeed, nfeed, nant, nant), we take an
    // outer product over feeds and antennas, contract over nax, and integrate over npix
    // (i.e. sum over pixels).

    // Get outta here if our thread is out of bounds
    if (row >= nant*nfeed || col >= nant*nfeed) return;

    uint ifeed = row / nant;
    uint jfeed = col / nant;
    uint iant  = row % nant;
    uint jant  = col % nant;

    {{ CDTYPE }} product_val = make_{{ CDTYPE }}(0.0, 0.0);
    {{ CDTYPE }} this_el;

    uint width = nfeed * nant;
    uint full_width = width * npix;
    uint iv, jv;
    for (int iax=0; iax<nax; iax++){
        for(int ipix=0; ipix<npix; ipix++) {
            iv = iax*full_width + ifeed*(nant*npix) + iant*npix + ipix;
            jv = iax*full_width + jfeed*(nant*npix) + jant*npix + ipix;
            this_el = cuCmul{{ f }}(cuConj{{ f }}(v[iv]),v[jv]);
            product_val = cuCadd{{ f }}(product_val, this_el);
        }
    }
    uint el = ifeed*nfeed*nant*nant + jfeed*nant*nant + iant *nant + jant;
    vis[el] = product_val;
    __syncthreads();
}
