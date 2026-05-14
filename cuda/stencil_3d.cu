#include <stdio.h>
#include <assert.h>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int NUM_REPS = 100;
#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)
#define IN_TILE_DIM1 32
#define OUT_TILE_DIM1 (IN_TILE_DIM1 - 2)

void cpu_stencil(int* output, const int* input, int n) {
    for (int z = 1; z < n-1; z++) {
        for (int y = 1; y < n-1; y++) {
            for (int x = 1; x < n-1; x++) {
                output[z * n * n + y * n + x] = input[z * n * n + y * n + x] +
                    input[(z-1) * n * n + y * n + x] + input[(z+1) * n * n + y * n + x] +
                    input[z * n * n + (y-1) * n + x] + input[z * n * n + (y+1) * n + x] +
                    input[z * n * n + y * n + (x-1)] + input[z * n * n + y * n + (x+1)];
            }
        }
    }
}

__global__ void stencil(int* output, const int* input, int n) {
    int z = blockIdx.z * BLOCK_DIM + threadIdx.z;
    int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int x = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (z >= 1 && z < n-1 && y >= 1 && y < n-1 && x >= 1 && x < n-1) {
        output[z * n * n + y * n + x] = input[z * n * n + y * n + x] +
            input[(z-1) * n * n + y * n + x] + input[(z+1) * n * n + y * n + x] +
            input[z * n * n + (y-1) * n + x] + input[z * n * n + (y+1) * n + x] +
            input[z * n * n + y * n + (x-1)] + input[z * n * n + y * n + (x+1)];
    }
}

__global__ void mmTiledStencil(int *output, const int *input, int n) {
    // It's gonna skip over OUT_TILE_DIM elements, not IN_TILE_DIM elements.
    // blockIdx.x * OUT_TILE_DIM is the beginning of the output tile.
    // But the 1st thread is not going to start at the beginning of the output.
    // It's gonna start from 1 row, 1 column and 1 depth before that.
    int z = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ int in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (z >= 0 && z < n && y >= 0 && y < n && x >= 0 && x < n) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = input[z * n * n + y * n + x];
    }
    __syncthreads();

    if (z >= 1 && z < n-1 && y >= 1 && y < n-1 && x >= 1 && x < n-1) {
        if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1 &&
            threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1) {
            // Only threads that are responsible for computing the output tile will do the computation.
            // The threads that are responsible for loading the halo region will not do the computation.
            output[z * n * n + y * n + x] = in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x] +
                in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1];
        }
    }
}

__global__ void mmTiledCoarseningStencil(int *output, const int *input, int n) {
    int z_start = blockIdx.z * OUT_TILE_DIM1;
    int y = blockIdx.y * OUT_TILE_DIM1 + threadIdx.y - 1;
    int x = blockIdx.x * OUT_TILE_DIM1 + threadIdx.x - 1;

    __shared__ int in_prev_s[IN_TILE_DIM1][IN_TILE_DIM1];
    __shared__ int in_curr_s[IN_TILE_DIM1][IN_TILE_DIM1];
    __shared__ int in_next_s[IN_TILE_DIM1][IN_TILE_DIM1];
    if (z_start - 1 >= 0 && z_start - 1 < n && y >= 0 && y < n && x >= 0 && x < n) {
        in_prev_s[threadIdx.y][threadIdx.x] = input[(z_start - 1) * n * n + y * n + x];
    }
    if (z_start >= 0 && z_start < n && y >= 0 && y < n && x >= 0 && x < n) {
        in_curr_s[threadIdx.y][threadIdx.x] = input[z_start * n * n + y * n + x];
    }
    __syncthreads();

    for (int z = z_start; z < z_start + OUT_TILE_DIM1; ++z) {
        if (z + 1 >= 0 && z + 1 < n && y >= 0 && y < n && x >= 0 && x < n) {
            in_next_s[threadIdx.y][threadIdx.x] = input[(z + 1) * n * n + y * n + x];
        }
        __syncthreads();

        if (z >= 1 && z < n-1 && y >= 1 && y < n-1 && x >= 1 && x < n-1) {
            if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM1 - 1 &&
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM1 - 1 &&
                threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM1 - 1) {
                output[z * n * n + y * n + x] = in_curr_s[threadIdx.y][threadIdx.x] +
                    in_curr_s[threadIdx.y - 1][threadIdx.x] + in_curr_s[threadIdx.y + 1][threadIdx.x] +
                    in_curr_s[threadIdx.y][threadIdx.x - 1] + in_curr_s[threadIdx.y][threadIdx.x + 1] +
                    in_prev_s[threadIdx.y][threadIdx.x] + in_next_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();

        in_prev_s[threadIdx.y][threadIdx.x] = in_curr_s[threadIdx.y][threadIdx.x];
        in_curr_s[threadIdx.y][threadIdx.x] = in_next_s[threadIdx.y][threadIdx.x];
    }
}

__global__ void mmTiledCoarseningStencilV2(int *output, const int *input, int n) {
    int z_start = blockIdx.z * OUT_TILE_DIM1;
    int y = blockIdx.y * OUT_TILE_DIM1 + threadIdx.y - 1;
    int x = blockIdx.x * OUT_TILE_DIM1 + threadIdx.x - 1;

    int in_prev_z;
    __shared__ int in_curr_s[IN_TILE_DIM1][IN_TILE_DIM1];
    int in_next_z;
    if (z_start - 1 >= 0 && z_start - 1 < n && y >= 0 && y < n && x >= 0 && x < n) {
        in_prev_z = input[(z_start - 1) * n * n + y * n + x];
    }
    if (z_start >= 0 && z_start < n && y >= 0 && y < n && x >= 0 && x < n) {
        in_curr_s[threadIdx.y][threadIdx.x] = input[z_start * n * n + y * n + x];
    }
    __syncthreads();

    for (int z = z_start; z < z_start + OUT_TILE_DIM1; ++z) {
        if (z + 1 >= 0 && z + 1 < n && y >= 0 && y < n && x >= 0 && x < n) {
            in_next_z  = input[(z + 1) * n * n + y * n + x];
        }
        __syncthreads();

        if (z >= 1 && z < n-1 && y >= 1 && y < n-1 && x >= 1 && x < n-1) {
            if (threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM1 - 1 &&
                threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM1 - 1 &&
                threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM1 - 1) {
                output[z * n * n + y * n + x] = in_curr_s[threadIdx.y][threadIdx.x] +
                    in_curr_s[threadIdx.y - 1][threadIdx.x] + in_curr_s[threadIdx.y + 1][threadIdx.x] +
                    in_curr_s[threadIdx.y][threadIdx.x - 1] + in_curr_s[threadIdx.y][threadIdx.x + 1] +
                    in_prev_z + in_next_z;
            }
        }
        __syncthreads();

        in_prev_z = in_curr_s[threadIdx.y][threadIdx.x];
        in_curr_s[threadIdx.y][threadIdx.x] = in_next_z;
    }
}

bool verify_result(const int* result, const int* expected, int n) {
    for (int i = 0; i < n; i++) {
        if (result[i] != expected[i]) {
            fprintf(stderr, "Mismatch at index %d: got %d, expected %d\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int N = 1 << 9;
    int n = N * N * N;

    auto input_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto output_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* input = input_buf.data();
    int* output = output_buf.data();
    int* expected = expected_buf.data();

    auto d_input_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto d_output_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* d_input = d_input_buf.data();
    int* d_output = d_output_buf.data();

    // Initialize input
    for (int i = 0; i < n; ++i) {
        input[i] = i % 10;
    }

    CUDA_OK(cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice));

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- Convolution using naive kernel --------------
    timer.start();
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (N + BLOCK_DIM - 1) / BLOCK_DIM);

    for (int i = 0; i < NUM_REPS; i++) {
        // Call a GPU kenrel function (launch a grid of threads)
        stencil<<<dimGrid, dimBlock>>>(d_output, d_input, N);
    }
    ms = timer.elapsed_ms();
    printf("[baseline] Average time per stencil: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    cpu_stencil(expected, input, N);
    assert(verify_result(output, expected, n));

    // -------------- Convolution using shared memory --------------
    timer.start();
    // Eatch block needs to have enough threads to process the entire input tile dimension
    dim3 dimBlock1(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    // We need enough blocks to cover all the output tiles
    dim3 dimGrid1((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                  (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                  (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    for (int i = 0; i < NUM_REPS; i++) {
        // Call a GPU kenrel function (launch a grid of threads)
        mmTiledStencil<<<dimGrid1, dimBlock1>>>(d_output, d_input, N);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per stencil: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(output, expected, n));

    // -------------- Convolution using shared memory and thread coarsening --------------
    timer.start();
    dim3 dimBlock2(IN_TILE_DIM1, IN_TILE_DIM1, 1);
    dim3 dimGrid2((N + OUT_TILE_DIM1 - 1) / OUT_TILE_DIM1,
                  (N + OUT_TILE_DIM1 - 1) / OUT_TILE_DIM1,
                  (N + OUT_TILE_DIM1 - 1) / OUT_TILE_DIM1);

    for (int i = 0; i < NUM_REPS; i++) {
        mmTiledCoarseningStencil<<<dimGrid2, dimBlock2>>>(d_output, d_input, N);
    }
    ms = timer.elapsed_ms();
    printf("[thread coarsening] Average time per stencil: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(output, expected, n));

    // -------------- Convolution using shared memory and thread coarsening v2 --------------
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        mmTiledCoarseningStencilV2<<<dimGrid2, dimBlock2>>>(d_output, d_input, N);
    }
    ms = timer.elapsed_ms();
    printf("[register coarsening] Average time per stencil: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(output, expected, n));

    printf("Stencil completed successfully.\n");

    return 0;
}
