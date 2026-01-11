#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

const int NUM_REPS = 100;
const int GRID_SIZE = 32;
const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void sum(int *output, const int *input, const int count) {
    __shared__ int s_mem[BLOCK_SIZE];

    // grid stride loop to load data
    s_mem[threadIdx.x] = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        s_mem[threadIdx.x] += input[i];
    }

    for (int total = blockDim.x / 2; total > 0; total >>= 1) {
        __syncthreads();
        if (threadIdx.x < total) { // parallel sweep reduction
            s_mem[threadIdx.x] += s_mem[threadIdx.x + total];
        }
    }

    // add the sum of all blocks together
    if (threadIdx.x == 0) {
        atomicAdd(output, s_mem[0]);
    }
}

__global__ void sumWarp(int *output, const int *input, const int count) {
    // grid stride loop to load data
    int val = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        val += input[i];
    }

    // first warp-shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ int s_mem[32];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // put warp results into shared memory
    if (laneId == 0) {
        s_mem[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        // reload val from shared memory if warp existed
        int warpCount = blockDim.x / WARP_SIZE;
        val = (laneId < warpCount) ? s_mem[laneId] : 0;

        // final warp-shuffle reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (laneId == 0) {    
            atomicAdd(output, val);
        }
    }
}

int main() {
    const int n = 1 << 16;
    const size_t size = n * sizeof(int);

    int *h_input = (int*)malloc(size);
    int *h_output = (int*)malloc(sizeof(int));

    int *d_input, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, size));
    CUDA_OK(cudaMalloc((void**)&d_output, size));

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_input[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- Transpose using naive kernel --------------
    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        cudaMemset(d_output, 0, sizeof(int));
        sum<<<GRID_SIZE, BLOCK_SIZE>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[baseline] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    int s = 0;
    for (int i = 0; i < n; ++i) {
        s += i;
    }
    assert(*h_output == s);

    // -------------- Transpose using warp-shuffle kernel --------------

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        cudaMemset(d_output, 0, sizeof(int));
        sumWarp<<<GRID_SIZE, BLOCK_SIZE>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[warp-shuffle] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(*h_output == s);

    printf("Reduction completed successfully.\n");

    free(h_input);
    free(h_output);
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_output));

    return 0;
}
