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

// Compute A*I*exp(ij*tau*freq) for all antennas, storing output in v
__global__ void MeasEq(
    {{ CDTYPE }} *A,
    {{ DTYPE }} *sqrtI,
    {{ DTYPE }} *tau,
    double freq,
    uint nsrc,
    uint *beam_idx,
    {{ CDTYPE }} *v
){
    const uint nbeam = {{ NBEAM }};
    const uint nax  = {{ NAX }};
    const uint nfeed= {{ NFEED }};
    const uint nant = {{ NANT }};

    const uint src  =  blockIdx.x * blockDim.x + threadIdx.x;  // first thread dim is src on sky
    const uint antax = blockIdx.y * blockDim.y + threadIdx.y;  // second thread dim is nax*nant
    const uint feed  = blockIdx.z * blockDim.z + threadIdx.z;  // third thread dim is nfeed

    uint ant = antax / nax;
    uint ax = antax % nax;
    uint beam = beam_idx[ant];

    {{ CDTYPE }} amp;
    {{ DTYPE }} phs;
    if (ant >= nant || ax >= nax || feed >= nfeed || src >= nsrc) return;

    // Create both real/imag parts of the "amplitude"
    uint tau_indx = ant*nsrc + src;
    uint Aindx = ax*(nfeed*nbeam*nsrc) + feed*nbeam*nsrc + beam*nsrc + src;
    amp.x = A[Aindx].x * sqrtI[src];
    amp.y = A[Aindx].y * sqrtI[src];

    phs = tau[tau_indx] * freq;

    uint vidx = feed*(nant*nax*nsrc) + ant*nax*nsrc + ax*nsrc + src;
    v[vidx] = cuCmul{{ f }}(amp, make_{{ CDTYPE }}(cos(phs), sin(phs)));

    __syncthreads(); // make sure everyone used mem before kicking out
}
