// Order-preserving stream compaction of the sources above the horizon.
//
// Replaces the ``cp.where(topo[2] > 0)`` horizon cut in
// matvis.core.coords.CoordinateRotation.select_chunk, whose result size is
// needed on the host and therefore synchronizes the stream once per chunk.
//
// The counting half is batched over *all* chunks of an integration, so the one
// remaining host readback happens once per integration rather than once per
// chunk; the scatter is then a single kernel per chunk with no readback at all.
//
// The compaction is a segmented prefix sum rather than an atomic counter, so
// the surviving sources keep their original order and the result is
// bit-for-bit identical to the cp.where path (and reproducible run to run).
//
//   1. hc_count_*   -- grid (HC_NBLOCKS, nchunks): each block counts the hits
//                      in its own contiguous segment of one chunk.
//   2. hc_scan      -- grid (nchunks,): one block per chunk exclusive-scans
//                      that chunk's HC_NBLOCKS counts and writes the total.
//   3. hc_compact_* -- grid (HC_NBLOCKS,), once per chunk: re-scan the segment,
//                      scatter the survivors, pad the tail.
//
// Used by matvis.core.coords.CoordinateRotation.

#define HC_BLOCK 256
#define HC_NBLOCKS 256

// Shared-memory inclusive scan over HC_BLOCK entries of s[]. Returns this
// thread's inclusive value; s[HC_BLOCK-1] holds the total afterwards.
__device__ __forceinline__ int hc_block_inclusive_scan(int* s)
{
    for (int off = 1; off < HC_BLOCK; off <<= 1) {
        const int v = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0;
        __syncthreads();
        s[threadIdx.x] += v;
        __syncthreads();
    }
    return s[threadIdx.x];
}

// Pass 1: per-block population count of (z > 0), for every chunk at once.
template <typename R>
__device__ void hc_count(
    const R* __restrict__ z,       // row 2 of all_coords_topo, (nsrc,)
    const long long nsrc,
    const long long chunk_size,
    const long long seg,           // sources per block within a chunk
    int* __restrict__ block_counts)  // (nchunks, HC_NBLOCKS)
{
    const long long offset = (long long)blockIdx.y * chunk_size;
    const long long n = min(chunk_size, nsrc - offset);
    const long long start = (long long)blockIdx.x * seg;
    const long long end = min(start + seg, n);

    int cnt = 0;
    for (long long i = start + threadIdx.x; i < end; i += HC_BLOCK) {
        cnt += (z[offset + i] > (R)0) ? 1 : 0;
    }

    __shared__ int s[HC_BLOCK];
    s[threadIdx.x] = cnt;
    __syncthreads();
    for (int off = HC_BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_counts[(long long)blockIdx.y * HC_NBLOCKS + blockIdx.x] = s[0];
    }
}

// Pass 3: scatter one chunk's survivors, then pad the tail.
//
// The tail is padded with the zenith direction (0, 0, 1) and zero flux rather
// than being left holding the previous chunk's values: zenith is a valid
// direction for the beam interpolator (so the t == 0 finiteness check still
// means something), and the zero flux is what makes those slots contribute
// nothing to Z.
template <typename R>
__device__ void hc_compact(
    const R* __restrict__ topo,   // (3, topo_stride); this chunk starts at `offset`
    const long long topo_stride,
    const long long offset,
    const R* __restrict__ flux,   // (topo_stride,), same offset
    const long long n,            // sources in this chunk
    const long long seg,
    const int* __restrict__ block_offsets,  // (HC_NBLOCKS,), exclusive scan
    R* __restrict__ out_coords,   // (3, nsrc_alloc)
    R* __restrict__ out_flux,     // (nsrc_alloc,)
    const long long nsrc_alloc,
    const long long total)        // sources above the horizon in this chunk
{
    const R* zrow = topo + 2 * topo_stride + offset;
    const long long start = (long long)blockIdx.x * seg;
    const long long end = min(start + seg, n);

    __shared__ int s[HC_BLOCK];
    int running = block_offsets[blockIdx.x];

    for (long long base = start; base < end; base += HC_BLOCK) {
        const long long i = base + threadIdx.x;
        const int keep = (i < end && zrow[i] > (R)0) ? 1 : 0;
        s[threadIdx.x] = keep;
        __syncthreads();

        const int incl = hc_block_inclusive_scan(s);
        const int pos = running + incl - keep;
        const int tile_total = s[HC_BLOCK - 1];

        if (keep && (long long)pos < nsrc_alloc) {
            out_coords[pos] = topo[offset + i];
            out_coords[nsrc_alloc + pos] = topo[topo_stride + offset + i];
            out_coords[2 * nsrc_alloc + pos] = zrow[i];
            out_flux[pos] = flux[offset + i];
        }
        __syncthreads();
        running += tile_total;
    }

    // Pad [total, nsrc_alloc) across the whole grid. Disjoint from the scatter
    // above, which only ever writes below `total`.
    const long long stride = (long long)HC_NBLOCKS * HC_BLOCK;
    for (long long i = total + blockIdx.x * (long long)HC_BLOCK + threadIdx.x;
         i < nsrc_alloc; i += stride) {
        out_coords[i] = (R)0;
        out_coords[nsrc_alloc + i] = (R)0;
        out_coords[2 * nsrc_alloc + i] = (R)1;
        out_flux[i] = (R)0;
    }
}

extern "C" {

// Pass 2: one block per chunk. Exclusive-scans that chunk's block counts in
// place and writes the chunk total.
__global__ void hc_scan(
    int* __restrict__ block_counts,  // (nchunks, HC_NBLOCKS)
    int* __restrict__ counts)        // (nchunks,)
{
    int* row = block_counts + (long long)blockIdx.x * HC_NBLOCKS;

    __shared__ int s[HC_BLOCK];
    const int own = row[threadIdx.x];
    s[threadIdx.x] = own;
    __syncthreads();

    const int incl = hc_block_inclusive_scan(s);
    row[threadIdx.x] = incl - own;

    if (threadIdx.x == HC_BLOCK - 1) counts[blockIdx.x] = incl;
}

__global__ void hc_count_f32(
    const float* z, long long nsrc, long long chunk_size, long long seg, int* bc)
{ hc_count<float>(z, nsrc, chunk_size, seg, bc); }

__global__ void hc_count_f64(
    const double* z, long long nsrc, long long chunk_size, long long seg, int* bc)
{ hc_count<double>(z, nsrc, chunk_size, seg, bc); }

__global__ void hc_compact_f32(
    const float* topo, long long topo_stride, long long offset, const float* flux,
    long long n, long long seg, const int* block_offsets,
    float* out_coords, float* out_flux, long long nsrc_alloc, long long total)
{ hc_compact<float>(topo, topo_stride, offset, flux, n, seg, block_offsets,
                    out_coords, out_flux, nsrc_alloc, total); }

__global__ void hc_compact_f64(
    const double* topo, long long topo_stride, long long offset, const double* flux,
    long long n, long long seg, const int* block_offsets,
    double* out_coords, double* out_flux, long long nsrc_alloc, long long total)
{ hc_compact<double>(topo, topo_stride, offset, flux, n, seg, block_offsets,
                     out_coords, out_flux, nsrc_alloc, total); }

}  // extern "C"
