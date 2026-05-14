#include <stdio.h>
#include <assert.h>
#include <cstring>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

int NUM_REPS = 100;

__host__ __device__ void merge_sequential(int* output, int* a, int* b, int m, int n) {
    unsigned int i = 0;
    unsigned int j = 0;
    unsigned int k = 0;

    while (i < m && j < n) { 
        if (a[i] < b[j]) {
            output[k++] = a[i++];
        } else {
            output[k++] = b[j++];
        }
    }

    while (i < m) {
        output[k++] = a[i++];
    }

    while (j < n) {
        output[k++] = b[j++];
    }
}

/// Find the co-rank of k in a and b, i.e., find the index i in a such that:
///   1. i + j = k, 0 <= i <= m , 0 <= j <=n => max(0, k-n) <= i <= min(m, k)
///   2. a[i - 1] <= b[j] and b[j - 1] <= a[i]
__device__ unsigned int co_rank(int* a, int* b, unsigned int m, unsigned int n, unsigned int k) {
    unsigned int iLow = k > n ? k - n : 0;
    unsigned int iHigh = k < m ? k : m;

    // Binary search to find the co-rank of k.
    while (true) {
        unsigned int i = iLow + (iHigh - iLow) / 2;
        unsigned int j = k - i;

        if (i > 0 && j < n && a[i - 1] > b[j]) {
            // i is too big.
            iHigh = i - 1;
        } else if (j > 0 && i < m && b[j - 1] > a[i]) {
            // i is too small.
            iLow = i + 1;
        } else {
            // We found the co-rank of k.
            return i;
        }
    }
}

#define ELEM_PER_THREAD 6
#define THREADS_PER_BLOCK 128
#define ELEM_PER_BLOCK (THREADS_PER_BLOCK * ELEM_PER_THREAD)

__global__ void merge(int* a, int* b, int* c, unsigned int m, unsigned int n) {
    unsigned int k = (blockIdx.x * blockDim.x + threadIdx.x) * ELEM_PER_THREAD;

    if (k >= m + n) {
        return;
    }

    unsigned int i = co_rank(a, b, m, n, k);
    unsigned int j = k - i;

    unsigned int kNext = k + ELEM_PER_THREAD < m + n ? k + ELEM_PER_THREAD : m + n;
    unsigned int iEnd = co_rank(a, b, m, n, kNext);
    unsigned int jEnd = kNext - iEnd;

    merge_sequential(c + k, a + i, b + j, iEnd - i, jEnd - j);
}

__global__ void merge_tiling(int* c, int* a, int* b, int m, int n) {
    // Find the block's segments
    unsigned int kBlock = blockIdx.x * ELEM_PER_BLOCK;
    unsigned int kNextBlock = blockIdx.x < gridDim.x - 1 ? kBlock + ELEM_PER_BLOCK : m + n;

    // One thread finds the co-rank of the block's segments,
    // and all threads in the block synchronize to get the segments.
    // Declare iBlock and iNextBlock inside of shared memory.
    __shared__ unsigned int iBlock;
    __shared__ unsigned int iNextBlock;
    if (threadIdx.x == 0) {
        iBlock = co_rank(a, b, m, n, kBlock);
        iNextBlock = co_rank(a, b, m, n, kNextBlock);
    }
    __syncthreads();

    unsigned int jBlock = kBlock - iBlock;
    unsigned int jNextBlock = kNextBlock - iNextBlock;

    // Load the block's segments into shared memory.
    __shared__ int a_s[ELEM_PER_BLOCK];
    unsigned int mBlock = iNextBlock - iBlock;
    for (unsigned int i = threadIdx.x; i < mBlock; i += blockDim.x) {
        a_s[i] = a[iBlock + i];
    }
    int* b_s = a_s + mBlock;
    unsigned int nBlock = jNextBlock - jBlock;
    for (unsigned int i = threadIdx.x; i < nBlock; i += blockDim.x) {
        b_s[i] = b[jBlock + i];
    }
    __syncthreads();

    // Merge in shared memory.
    __shared__ int c_s[ELEM_PER_BLOCK];
    unsigned int k = threadIdx.x * ELEM_PER_THREAD;
    if (k >= mBlock + nBlock) {
        return;
    }
    unsigned int i = co_rank(a_s, b_s, mBlock, nBlock, k);
    unsigned int j = k - i;
    unsigned int kNext = k + ELEM_PER_THREAD < mBlock + nBlock ? k + ELEM_PER_THREAD : mBlock + nBlock;
    unsigned int iEnd = co_rank(a_s, b_s, mBlock, nBlock, kNext);
    unsigned int jEnd = kNext - iEnd;
    merge_sequential(c_s + k, a_s + i, b_s + j, iEnd - i, jEnd - j);
    __syncthreads();

    // Write the block's merged segment to global memory.
    for (unsigned int i = threadIdx.x; i < mBlock + nBlock; i += blockDim.x) {
        c[kBlock + i] = c_s[i];
    }
}

bool verify_result(int* result, int* expected, int n) {
    for (int i = 0; i < n; i++) {
        if (result[i] != expected[i]) {
            fprintf(stderr, "Mismatch at index %d: got %d, expected %d\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main() {
    unsigned int m = 1 << 6;
    unsigned int n = 1 << 6;

    auto a_buf = HostBuffer<int>::with_capacity(m);
    auto b_buf = HostBuffer<int>::with_capacity(n);
    auto c_buf = HostBuffer<int>::with_capacity(m + n);
    auto expected_buf = HostBuffer<int>::with_capacity(m + n);
    int* a = a_buf.data();
    int* b = b_buf.data();
    int* c = c_buf.data();
    int* expected = expected_buf.data();

    auto d_a_buf = CudaBuffer<int>::with_capacity(m);
    auto d_b_buf = CudaBuffer<int>::with_capacity(n);
    auto d_c_buf = CudaBuffer<int>::with_capacity(m + n);
    int* d_a = d_a_buf.data();
    int* d_b = d_b_buf.data();
    int* d_c = d_c_buf.data();

    // Initialize input
    for (int i = 0; i < m; ++i) {
        a[i] = i % 2000;
    }
    for (int i = 0; i < n; ++i) {
        b[i] = i % 1000;
    }

    merge_sequential(expected, a, b, m, n);

    d_a_buf.copy_from_host(a, static_cast<std::size_t>(m));
    d_b_buf.copy_from_host(b, static_cast<std::size_t>(n));

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- Merge using naive kernel --------------
    timer.start();
    unsigned int numBlocks = (m + n + ELEM_PER_BLOCK - 1) / ELEM_PER_BLOCK;

    for (int i = 0; i < NUM_REPS; i++) {
        merge<<<numBlocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, m, n);
    }
    ms = timer.elapsed_ms();
    printf("[baseline] Average time per merge: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    d_c_buf.copy_to_host(c, static_cast<std::size_t>(m + n));
    d_c_buf.synchronize();

    // Verify result
    assert(verify_result(c, expected, m + n));

    // -------------- Merge using shared memory --------------
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        merge_tiling<<<numBlocks, THREADS_PER_BLOCK>>>(d_c, d_a, d_b, m, n);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per merge: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    d_c_buf.copy_to_host(c, static_cast<std::size_t>(m + n));
    d_c_buf.synchronize();

    // Verify result
    assert(verify_result(c, expected, m + n));

    printf("Merge completed successfully.\n");

    return 0;
}
