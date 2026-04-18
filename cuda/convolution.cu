#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

const int NUM_REPS = 100;
#define OUT_TILE_DIM 32
#define MASK_RADIUS 2
#define MASK_DIM ((MASK_RADIUS) * 2 + 1)
#define IN_TILE_DIM (OUT_TILE_DIM + MASK_DIM - 1)

__constant__ int mask_c[MASK_DIM][MASK_DIM];

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

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

    int* input = (int*)malloc(n * sizeof(int));
    int* output = (int*)malloc(n * sizeof(int));

    int* d_input, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, n * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_output, n * sizeof(int)));

    // Initialize input
    for (int i = 0; i < n; ++i) {
        input[i] = i % 100;
    }

    CUDA_OK(cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice));

    // Copy mask to constant memory
    CUDA_OK(cudaMemcpyToSymbol(mask_c, mask, MASK_DIM * MASK_DIM * sizeof(int)));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- Convolution using naive kernel --------------
    CUDA_OK(cudaEventRecord(startEvent));
    dim3 dimBlock(OUT_TILE_DIM, OUT_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    for (int i = 0; i < NUM_REPS; i++) {
        // Call a GPU kenrel function (launch a grid of threads)
        convolution<<<dimGrid, dimBlock>>>(d_output, mask_c, d_input, width, height);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[baseline] Average time per convolution: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    int* expected = (int*)malloc(n * sizeof(int));
    cpu_convolution(expected, mask, input, width, height);
    assert(verify_result(output, expected, n));

    // -------------- Convolution using shared memory --------------
    CUDA_OK(cudaEventRecord(startEvent));
    dim3 dimBlock1(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid1((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    for (int i = 0; i < NUM_REPS; i++) {
        // Call a GPU kenrel function (launch a grid of threads)
        mmTiledConvolution<<<dimGrid1, dimBlock1>>>(d_output, mask_c, d_input, width, height);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[shared memory] Average time per convolution: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(output, expected, n));

    printf("Convolution completed successfully.\n");

    free(input);
    free(output);
    free(expected);
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_output));

    return 0;
}
