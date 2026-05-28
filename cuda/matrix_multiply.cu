#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cuda_fp16.h>
#include <mma.h>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int NUM_REPS = 100;

void cpu_matrix_multiply(int *output, const int *a, const int *b, int m, int n, int k) {
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < k; x++) {
            int tmp = 0;
            for (int step = 0; step < n; step++) {
                tmp += a[y * n + step] * b[step * k + x];
            }
            output[y * k + x] = tmp;
        }
    }
}

void cpu_matrix_multiply_float(float *output, const float *a, const float *b, int m, int n, int k) {
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < k; x++) {
            float tmp = 0.0f;
            for (int step = 0; step < n; step++) {
                tmp += a[y * n + step] * b[step * k + x];
            }
            output[y * k + x] = tmp;
        }
    }
}

__global__ void multiply_naive(int *output, const int *a, const int *b, int m, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m && y < k) {
        int tmp = 0;
        for (int step = 0; step < n; step++) {
            tmp += a[y * n + step] * b[step * k + x];
        }
        output[y * k + x] = tmp;
    }
}

// ref https://blog.csdn.net/kunhe0512/article/details/131381155
// a[][] * b[][] = c[][]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31
// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
template <int BLOCK_SIZE>
__global__ void multiply_tiling(int *output, const int *a, const int *b, int m, int n, int k) {
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = 0;

    for (int step = 0; step <= n; step += blockDim.x) {
        int step_y = y;
        int step_x = step + threadIdx.x;
        if (step_y < m && step_x < n) {
            sub_a[threadIdx.y][threadIdx.x] = a[step_y * n + step_x];
        } else {
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }

        step_y = step + threadIdx.y;
        step_x = x;
        if (step_y < n && step_x < k) {
            sub_b[threadIdx.y][threadIdx.x] = b[step_y * k + step_x];
        } else {
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; i++) {
            tmp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (y < m && x < k) {
        output[y * k + x] = tmp;
    }
}

template <int BLOCK_SIZE>
// Vectorized tiled GEMM kernel: each thread computes 4 contiguous output columns.
__global__ void multiply_tiling_int4(int *output, const int *a, const int *b, int m, int n, int k) {
    static_assert(BLOCK_SIZE % 4 == 0, "BLOCK_SIZE must be divisible by 4");
	// Keep A tile scalar: each iteration consumes one A value and broadcasts it across 4 B lanes.
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    // B tile is stored as int4 vectors along the column dimension.
    __shared__ int4 sub_b[BLOCK_SIZE][BLOCK_SIZE / 4];

    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // threadIdx.x indexes vector lanes, each lane covers 4 scalar columns.
    int vec_x = threadIdx.x;
    int x = blockIdx.x * BLOCK_SIZE + vec_x * 4;

    int4 sum = make_int4(0, 0, 0, 0);

    for (int step = 0; step < n; step += BLOCK_SIZE) {
        int a_col = step + vec_x * 4;
        int *a_dst = &sub_a[threadIdx.y][vec_x * 4];

        // Vectorized load from A when 4-wide access is in-bounds.
        if (y < m && (a_col + 3) < n) {
			// The addresses of the shared memory are totally different than the addresses of the global memory.
			// Total in the size of each address, for example, for the shard memory,
			// we have six hexadecimal digits size.
			// For the global memory, it has like 12 hexadecimal digits.
            *reinterpret_cast<int4 *>(a_dst) =
                *reinterpret_cast<const int4 *>(&a[y * n + a_col]);
        } else {
            a_dst[0] = (y < m && a_col < n) ? a[y * n + a_col] : 0;
            a_dst[1] = (y < m && (a_col + 1) < n) ? a[y * n + a_col + 1] : 0;
            a_dst[2] = (y < m && (a_col + 2) < n) ? a[y * n + a_col + 2] : 0;
            a_dst[3] = (y < m && (a_col + 3) < n) ? a[y * n + a_col + 3] : 0;
        }

        int b_row = step + threadIdx.y;
        // Vectorized load from B when 4 contiguous columns are in-bounds.
        if (b_row < n && (x + 3) < k) {
            sub_b[threadIdx.y][vec_x] =
                *reinterpret_cast<const int4 *>(&b[b_row * k + x]);
        } else {
            // Scalar fallback for boundary tiles.
            int4 b_vec = make_int4(0, 0, 0, 0);
            if (b_row < n && x < k) b_vec.x = b[b_row * k + x];
            if (b_row < n && (x + 1) < k) b_vec.y = b[b_row * k + x + 1];
            if (b_row < n && (x + 2) < k) b_vec.z = b[b_row * k + x + 2];
            if (b_row < n && (x + 3) < k) b_vec.w = b[b_row * k + x + 3];
            sub_b[threadIdx.y][vec_x] = b_vec;
        }

        __syncthreads();

        // Dot product between one A row fragment and one B column-vector fragment.
        for (int i = 0; i < BLOCK_SIZE; i++) {
			// C[y, x] = Σ_i A[y, i] * B[i, x]
			// A thread not only computes C[y, x], but also computes:
			// - C[y, x]
			// - C[y, x+1]
			// - C[y, x+2]
			// - C[y, x+3]
			int a_val = sub_a[threadIdx.y][i];
            int4 b_vec = sub_b[i][vec_x];
            sum.x += a_val * b_vec.x;
            sum.y += a_val * b_vec.y;
            sum.z += a_val * b_vec.z;
            sum.w += a_val * b_vec.w;
        }
        __syncthreads();
    }

    if (y < m && x < k) output[y * k + x] = sum.x;
    if (y < m && (x + 1) < k) output[y * k + x + 1] = sum.y;
    if (y < m && (x + 2) < k) output[y * k + x + 2] = sum.z;
    if (y < m && (x + 3) < k) output[y * k + x + 3] = sum.w;
}

template <int BLOCK_SIZE, int COARSE_FACTOR>
__global__ void multiply_tiling_coarse(int *output, const int *a, const int *b, int m, int n, int k) {
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;
    int sum[COARSE_FACTOR];
    for (int i = 0; i < COARSE_FACTOR; i++) {
        sum[i] = 0;
    }

    for (int step = 0; step <= n; step += blockDim.x) {
        int step_y = y;
        int step_x = step + threadIdx.x;
        if (step_y < m && step_x < n) {
            sub_a[threadIdx.y][threadIdx.x] = a[step_y * n + step_x];
        } else {
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }

        for (int c = 0; c < COARSE_FACTOR; c++) {
            unsigned int col = xStart + c * BLOCK_SIZE;
            step_y = step + threadIdx.y;
            step_x = col;
            if (step_y < n && step_x < k) {
                sub_b[threadIdx.y][threadIdx.x] = b[step_y * k + step_x];
            } else {
                sub_b[threadIdx.y][threadIdx.x] = 0;
            }
            __syncthreads();

            for (int i = 0; i < blockDim.x; i++) {
                sum[c] += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
            }
            __syncthreads();
        }
    }

    for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
        unsigned int col = xStart + i * BLOCK_SIZE;
        if (y < m && col < k) {
            output[y * k + col] = sum[i];
        }
    }
}

// WMMA uses tensor cores on 16x16x16 fragments. The input matrices use half
// precision because tensor cores consume FP16 operands efficiently, while the
// accumulator stays in float to reduce round-off error.
//
// Each CUDA block owns one 16x16 output tile C[tile_row:tile_row+16,
// tile_col:tile_col+16]. This compact teaching kernel assumes m, n, and k are
// multiples of 16; pad inputs first if boundary tiles are needed.
__global__ void multiply_tiling_wmma(float *output, const half *a, const half *b, int m, int n, int k) {
    using namespace nvcuda;

    // Map the block index to the top-left corner of the output tile.
    int tile_row = blockIdx.y * 16;
    int tile_col = blockIdx.x * 16;

    if (tile_row >= m || tile_col >= k) {
        return;
    }

    // Fragments are warp-level register tiles managed by the WMMA API.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Walk across the K dimension in 16-wide slices:
    // C_tile += A_tile(row, tile_k) * B_tile(tile_k, col).
    for (int tile_k = 0; tile_k < n; tile_k += 16) {
        wmma::load_matrix_sync(a_frag, a + tile_row * n + tile_k, n);
        wmma::load_matrix_sync(b_frag, b + tile_k * k + tile_col, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the completed 16x16 tile back to row-major global memory.
    wmma::store_matrix_sync(output + tile_row * k + tile_col, c_frag, k, wmma::mem_row_major);
}

bool verify_result(const int *a, const int *b, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (a[y * width + x] != b[y * width + x]) {
                return false;
            }
        }
    }
    return true;
}

bool verify_result_float(const float *a, const float *b, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float diff = fabsf(a[y * width + x] - b[y * width + x]);
            if (diff > 1e-3f) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const int *matrix, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%d ", matrix[y * width + x]);
        }
        printf("\n");
    }
}

// nvcc matrix_multiply.cu -arch=sm_70
int main() {
    int m = 1 << 10;
    int n = 1 << 11;
    int k = 1 << 10;

    auto h_a_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(m) * n);
    auto h_b_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n) * k);
    auto h_c_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(m) * k);
    auto c_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(m) * k);
    // WMMA path uses separate buffers so the original integer examples stay unchanged.
    auto h_a_wmma_buf = HostBuffer<half>::with_capacity(static_cast<std::size_t>(m) * n);
    auto h_b_wmma_buf = HostBuffer<half>::with_capacity(static_cast<std::size_t>(n) * k);
    // Keep float copies on the host to build a CPU reference result before converting to half.
    auto h_a_wmma_float_buf = HostBuffer<float>::with_capacity(static_cast<std::size_t>(m) * n);
    auto h_b_wmma_float_buf = HostBuffer<float>::with_capacity(static_cast<std::size_t>(n) * k);
    auto h_c_wmma_buf = HostBuffer<float>::with_capacity(static_cast<std::size_t>(m) * k);
    auto c_wmma_buf = HostBuffer<float>::with_capacity(static_cast<std::size_t>(m) * k);
    int* h_a = h_a_buf.data();
    int* h_b = h_b_buf.data();
    int* h_c = h_c_buf.data();
    int* c = c_buf.data();
    half* h_a_wmma = h_a_wmma_buf.data();
    half* h_b_wmma = h_b_wmma_buf.data();
    float* h_a_wmma_float = h_a_wmma_float_buf.data();
    float* h_b_wmma_float = h_b_wmma_float_buf.data();
    float* h_c_wmma = h_c_wmma_buf.data();
    float* c_wmma = c_wmma_buf.data();

    auto d_a_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(m) * n);
    auto d_b_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n) * k);
    auto d_c_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(m) * k);
    auto d_a_wmma_buf = CudaBuffer<half>::with_capacity(static_cast<std::size_t>(m) * n);
    auto d_b_wmma_buf = CudaBuffer<half>::with_capacity(static_cast<std::size_t>(n) * k);
    auto d_c_wmma_buf = CudaBuffer<float>::with_capacity(static_cast<std::size_t>(m) * k);
    int* d_a = d_a_buf.data();
    int* d_b = d_b_buf.data();
    int* d_c = d_c_buf.data();
    half* d_a_wmma = d_a_wmma_buf.data();
    half* d_b_wmma = d_b_wmma_buf.data();
    float* d_c_wmma = d_c_wmma_buf.data();

    // Initialize input matrix
    for (int i = 0; i < m * n; ++i) {
        h_a[i] = i;
    }
    for (int i = 0; i < n * k; ++i) {
        h_b[i] = i;
    }
    // Use small integer-valued floats so half conversion is exact and CPU/GPU
    // comparison only reflects accumulation behavior.
    for (int i = 0; i < m * n; ++i) {
        h_a_wmma_float[i] = static_cast<float>(i % 3);
        h_a_wmma[i] = __float2half(h_a_wmma_float[i]);
    }
    for (int i = 0; i < n * k; ++i) {
        h_b_wmma_float[i] = static_cast<float>(i % 5);
        h_b_wmma[i] = __float2half(h_b_wmma_float[i]);
    }

    d_a_buf.copy_from_host(h_a, static_cast<std::size_t>(m) * n);
    d_b_buf.copy_from_host(h_b, static_cast<std::size_t>(n) * k);
    d_a_wmma_buf.copy_from_host(h_a_wmma, static_cast<std::size_t>(m) * n);
    d_b_wmma_buf.copy_from_host(h_b_wmma, static_cast<std::size_t>(n) * k);

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- Multiplication using naive kernel --------------

    const int block_size = 32;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        multiply_naive<<<dimGrid, dimBlock>>>(d_c, d_a, d_b, m, n, k);
    }
    ms = timer.elapsed_ms();
    printf("[native] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    d_c_buf.copy_to_host(h_c, static_cast<std::size_t>(m) * k);
    d_c_buf.synchronize();

    cpu_matrix_multiply(c, h_a, h_b, m, n, k);
    assert(verify_result(h_c, c, m, k));

    // -------------- Multiplication using shared memory kernel --------------

    const int BLOCK_SIZE = 32;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        multiply_tiling<BLOCK_SIZE><<<gridSize, blockSize>>>(d_c, d_a, d_b, m, n, k);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    d_c_buf.copy_to_host(h_c, static_cast<std::size_t>(m) * k);
    d_c_buf.synchronize();

    assert(verify_result(h_c, c, m, k));

    // -------------- Multiplication using WMMA tensor core kernel --------------

    // Build the CPU reference using the pre-conversion float inputs.
    cpu_matrix_multiply_float(c_wmma, h_a_wmma_float, h_b_wmma_float, m, n, k);

    // One warp computes one 16x16 WMMA output tile.
    dim3 blockSizeWmma(32);
    dim3 gridSizeWmma((k + 15) / 16, (m + 15) / 16);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        multiply_tiling_wmma<<<gridSizeWmma, blockSizeWmma>>>(d_c_wmma, d_a_wmma, d_b_wmma, m, n, k);
    }
    ms = timer.elapsed_ms();
    printf("[wmma tensor core] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    d_c_wmma_buf.copy_to_host(h_c_wmma, static_cast<std::size_t>(m) * k);
    d_c_wmma_buf.synchronize();

    // WMMA accumulates in float, so a small tolerance is used instead of exact equality.
    assert(verify_result_float(h_c_wmma, c_wmma, k, m));

    // -------------- Multiplication using shared memory + int4 kernel --------------

    // int4 version: x-dimension threads are reduced by 4, each thread outputs 4 columns.

    dim3 blockSizeInt4(BLOCK_SIZE / 4, BLOCK_SIZE);
    dim3 gridSizeInt4((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        multiply_tiling_int4<BLOCK_SIZE><<<gridSizeInt4, blockSizeInt4>>>(d_c, d_a, d_b, m, n, k);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory + int4] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    d_c_buf.copy_to_host(h_c, static_cast<std::size_t>(m) * k);
    d_c_buf.synchronize();

    assert(verify_result(h_c, c, m, k));

    // -------------- Multiplication using shared memory and coarsening kernel --------------

    const int COARSE_FACTOR = 4;
    dim3 gridSize1((k + BLOCK_SIZE - 1) / BLOCK_SIZE / COARSE_FACTOR, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        multiply_tiling_coarse<BLOCK_SIZE, COARSE_FACTOR><<<gridSize1, blockSize>>>(d_c, d_a, d_b, m, n, k);
    }
    ms = timer.elapsed_ms();
    printf("[coarsening] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    d_c_buf.copy_to_host(h_c, static_cast<std::size_t>(m) * k);
    d_c_buf.synchronize();

    assert(verify_result(h_c, c, m, k));

    printf("Multiplication completed successfully.\n");

    return 0;
}
