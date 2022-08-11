/*
    Functions for interpolation of the beam onto source/pixel locations.

    See measurement_equation.c for available Jinja template parameters.
*/
#include <cuComplex.h>
#include <pycuda-helpers.hpp>
#include <stdio.h>

// __shared__ {{ DTYPE }} sh_buf[{{ BLOCK_PX }}*5];

// Linearly interpolate between [v0,v1] for t=[0,1]
// v = v0 * (1-t) + v1 * t = t*v1 + (-t*v0 + v0)
// Runs on GPU only
__device__ inline {{ DTYPE }} lerp({{ DTYPE }} v0, {{ DTYPE }} v1, {{ DTYPE }} t) {
    return fma(t, v1, fma(-t, v0, v0));
}

// 3D texture storing beam response on (x=sin th_x, y=sin th_y, nant) grid
// for fast lookup by multiple threads.  Suggest setting first 2 dims of
// bm_tex to an odd number to get pixel centered on zenith.  The pixels
// on the border are centered at [-1,1] respectively.  Note that this
// matrix is transposed relative to the host-side matrix used to set it.
texture<fp_tex_{{ DTYPE }}, cudaTextureType3D, cudaReadModeElementType> bm_tex;

// Interpolate bm_tex[x,y] at top=(x,y,z) coords and store answer in "A"
// __global__ void InterpolateBeam({{ DTYPE }} *top, {{ DTYPE }} *A)
// {
//     const uint nant = {{ NANT }};
//     const uint npix = {{ NPIX }};
//     const uint tx = threadIdx.x; // switched to make first dim px
//     const uint ty = threadIdx.y; // switched to make second dim ant
//     const uint pix = blockIdx.x * blockDim.x + threadIdx.x;
//     const uint ant = blockIdx.y * blockDim.y + threadIdx.y;

//     {{ DTYPE }} bm_x, bm_y, px, py, pz, fx, fy, top_z;
//     if (pix >= npix || ant >= nant) return;
//     if (ty == 0) // buffer top_z for all antennas
//         sh_buf[tx+{{ BLOCK_PX }} * 4] = top[2*npix+pix];
//     __syncthreads(); // make sure top_z exists for all threads
//     top_z = sh_buf[tx+{{ BLOCK_PX }} * 4];
//     if (ty == 0 && top_z > 0) { // buffer x interpolation for all threads
//         bm_x = (beam_px-1) * (0.5 * top[pix] + 0.5);
//         px = floorf(bm_x);
//         sh_buf[tx+{{ BLOCK_PX }} * 0] = bm_x - px; // fx, fractional position
//         sh_buf[tx+{{ BLOCK_PX }} * 2] = px + 0.5f; // px, pixel index
//     }
//     if (ty == 1 && top_z > 0) { // buffer y interpolation for all threads
//         bm_y = (beam_px-1) * (0.5 * top[npix+pix] + 0.5);
//         py = floorf(bm_y);
//         sh_buf[tx+{{ BLOCK_PX }} * 1] = bm_y - py; // fy, fractional position
//         sh_buf[tx+{{ BLOCK_PX }} * 3] = py + 0.5f; // py, pixel index
//     }
//     __syncthreads(); // make sure interpolation exists for all threads
//     if (top_z > 0) {
//         fx = sh_buf[tx+{{ BLOCK_PX }} * 0];
//         fy = sh_buf[tx+{{ BLOCK_PX }} * 1];
//         px = sh_buf[tx+{{ BLOCK_PX }} * 2];
//         py = sh_buf[tx+{{ BLOCK_PX }} * 3];
//         pz = ant + 0.5f;
//         A[ant*npix+pix] = lerp(
//             lerp(fp_tex3D(bm_tex,px,py,pz), fp_tex3D(bm_tex,px+1.0f,py,pz), fx),
//             lerp(fp_tex3D(bm_tex,px,py+1.0f,pz),
//                 fp_tex3D(bm_tex,px+1.0f,py+1.0f,pz),fx), fy);
//     } else {
//         A[ant*npix+pix] = 0;
//     }
//     __syncthreads(); // make sure everyone used mem before kicking out
// }


__global__ void InterpolateBeamAltAz(
    {{ DTYPE }} *az,
    {{ DTYPE }} *za,
    {{ DTYPE }} *Agrid,
    uint nsrc,
    {{ DTYPE }} *Asrc
){
    const uint nbeam = {{ NBEAM }};

    const {{ DTYPE }} daz = {{ DAZ }};
    const {{ DTYPE }} dza = {{ DZA }};
    const uint beam_n_az = {{ BEAM_N_AZ }};
    const uint beam_n_za = {{ BEAM_N_ZA }};

    const uint src = blockIdx.x * blockDim.x + threadIdx.x;
    const uint ant = blockIdx.y * blockDim.y + threadIdx.y;
    const uint pol = blockIdx.z * blockDim.z + threadIdx.z;  // stands for ax + feed

    uint npol = {{ NAX }} * {{ NFEED }};

    {{ DTYPE }} px, py, fx, fy;

    // Return early if our thread is out of range
    if (src >= nsrc || ant >= nbeam || pol >= npol) return;

    // Agrid has shape (Nbeam, Nax, Nfeed, Nza, Naz)
    // Asrc has shape  (Nax, Nfeed, Nbeam, Nsrc)

    // fx is the fraction of a pixel in which the azimuth is situated.
    // px is the pixel of the azimuth.
    // TODO: the following is the same at every ant and pol, so could probably be
    //       cached somehow...
    px = floorf(az[src] / daz);  // must be integer between [0, Naz-1]
    fx = (az[src] - px*daz)/daz;  // must be between [0, 1)

    py = floorf(za[src] / dza);
    fy = (za[src] - py*dza)/dza;


    // Find the index of the start of this bracket of Nza/Naz in Agrid.
    uint root = ant*npol * beam_n_az * beam_n_za + pol * beam_n_az * beam_n_za;

    // Get left-bottom corner of the square in which the src lives
    uint origin = root + py*beam_n_az + px;

    Asrc[pol * nbeam * nsrc + ant * nsrc + src] = lerp(
        lerp(Agrid[origin], Agrid[origin + 1], fx),
        lerp(Agrid[origin + beam_n_az], Agrid[origin + beam_n_az + 1], fx),
        fy
    );

//     if(ant==0){
//         printf("ant=%d root=%d nsrc=%d az=%f za=%f px=%f py=%f fx=%f fy=%f Agrid=[%f %f %f %f] Answer=%f\n ",
//             ant, root, nsrc, az[src], za[src], px, py, fx, fy,
//             Agrid[origin], Agrid[origin + 1], Agrid[origin + beam_n_az], Agrid[origin + beam_n_az + 1],
//             Asrc[pol * nbeam * nsrc + ant * nsrc + src]
//         );
//    }

    __syncthreads(); // make sure everyone used mem before kicking out
}
