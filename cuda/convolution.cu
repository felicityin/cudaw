#include <stdio.h>
#include <assert.h>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int NUM_REPS = 100;
#define OUT_TILE_DIM 32
#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS) * 2 + 1)
#define IN_TILE_DIM (OUT_TILE_DIM + MASK_DIM - 1)

__constant__ int mask_c[MASK_DIM][MASK_DIM];

void cpu_convolution(int* output, const int mask[MASK_DIM][MASK_DIM],
                     const int* input, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            for (int yy = 0; yy < MASK_DIM; yy++) {
                for (int xx = 0; xx < MASK_DIM; xx++) {
                    int y_ = y - MASK_RADIUS + yy;
                    int x_ = x - MASK_RADIUS + xx;
                    if (y_ >= 0 && y_ < height && x_ >= 0 && x_ < width) {
                        sum += mask[yy][xx] * input[y_ * width + x_];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

__global__ void convolution(int* output, const int mask[MASK_DIM][MASK_DIM],
                            const int* input, int width, int height) {
    int outRow = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int outCol = blockIdx.x * OUT_TILE_DIM + threadIdx.x;

    if (outRow < height && outCol < width) { 
        int sum = 0;
        for (int yy = 0; yy < MASK_DIM; yy++) {
            for (int xx = 0; xx < MASK_DIM; xx++) {
                int y_ = outRow - MASK_RADIUS + yy;
                int x_ = outCol - MASK_RADIUS + xx;
                if (y_ >= 0 && y_ < height && x_ >= 0 && x_ < width) {
                    sum += mask_c[yy][xx] * input[y_ * width + x_];
                }
            }
        }
        output[outRow * width + outCol] = sum;
    }
}

__global__ void mmTiledConvolution(int* output, const int mask[MASK_DIM][MASK_DIM],
                                     const int* input, int width, int height) {
    __shared__ int tiled_s[IN_TILE_DIM][IN_TILE_DIM];

    int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - MASK_DIM + 1;
    int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - MASK_DIM + 1;
    if (y >= 0 && y < height && x >= 0 && x < width) {
        tiled_s[threadIdx.y][threadIdx.x] = input[y * width + x];
    } else {
        tiled_s[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    y = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    x = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    if (y < height && x < width) {
        int sum = 0;
        for (int yy = 0; yy < MASK_DIM; yy++) {
            for (int xx = 0; xx < MASK_DIM; xx++) {
                int y_ = threadIdx.y + yy;
                int x_ = threadIdx.x + xx;
                if (y_ >= 0 && y_ < IN_TILE_DIM && x_ >= 0 && x_ < IN_TILE_DIM) {
                    sum += mask_c[yy][xx] * tiled_s[y_][x_];
                }
            }
        }
        output[y * width + x] = sum;
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
    int width = 1 << 11;
    int height = 1 << 10;
    int n = width * height;
    int mask[][MASK_DIM] = {
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4},
        {1, 2, 3, 4}
    };

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
        input[i] = i % 100;
    }

    CUDA_OK(cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice));

    // Copy mask to constant memory
    CUDA_OK(cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(int)));

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- Convolution using naive kernel --------------
    timer.start();
    dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    for (int i = 0; i < NUM_REPS; i++) {
        // Call a GPU kenrel function (launch a grid of threads)
        convolution<<<dimGrid, dimBlock>>>(d_output, mask_c, d_input, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[baseline] Average time per convolution: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    cpu_convolution(expected, mask, input, width, height);
    assert(verify_result(output, expected, n));

    // -------------- Convolution using shared memory --------------
    timer.start();
    dim3 dimBlock1(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid1((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    for (int i = 0; i < NUM_REPS; i++) {
        // Call a GPU kenrel function (launch a grid of threads)
        mmTiledConvolution<<<dimGrid1, dimBlock1>>>(d_output, mask_c, d_input, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per convolution: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(output, expected, n));

    printf("Convolution completed successfully.\n");

    return 0;
}
