#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

const int NUM_REPS = 100;
const int GRID_SIZE = 32;
const int BLOCK_SIZE = 256;
const int TOP_K = 10;

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// topk: 10, 9, 8, ..., 1
__device__ __host__ void insertTopK(int *topk, int val) {
    if (val <= topk[TOP_K - 1]) {
        return;
    }

    for (int i = TOP_K - 2; i >= 0; --i) {
        if (val > topk[i]) {
            topk[i + 1] = topk[i];
        } else {
            topk[i + 1] = val;
            return;
        }
    }

    topk[0] = val;
}

__global__ void topK(int *output, const int *input, const int len) {
    __shared__ int s_mem[BLOCK_SIZE * TOP_K];

    int topk[TOP_K];
    for (int i = 0; i < TOP_K; ++i) {
        topk[i] = INT_MIN;
    }

    // grid stride loop to load data
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < len; i += blockDim.x * gridDim.x) {
        insertTopK(topk, input[i]);
    }
    // write thread-local top K to shared memory
    for (int i = 0; i < TOP_K; ++i) {
        s_mem[threadIdx.x * TOP_K + i] = topk[i];
    }

    for (int total = blockDim.x / 2; total > 0; total >>= 1) {
        __syncthreads();
        if (threadIdx.x < total) { // parallel sweep reduction
            for (int i = 0; i < TOP_K; ++i) {
                insertTopK(&s_mem[threadIdx.x * TOP_K], s_mem[(threadIdx.x + total) * TOP_K + i]);
            }
        }
    }

    // put the top K of all blocks to the output
    if (threadIdx.x == 0) {
        for (int i = 0; i < TOP_K; ++i) {
            output[blockIdx.x * TOP_K + i] = s_mem[i];
        }
    }
}

int main() {
    const int n = 1 << 16;
    const size_t size = n * sizeof(int);

    int *h_input = (int*)malloc(size);
    int *h_output = (int*)malloc(sizeof(int) * TOP_K);

    int *d_input, *d_output_1, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, size));
    CUDA_OK(cudaMalloc((void**)&d_output_1, sizeof(int) * TOP_K * GRID_SIZE));
    CUDA_OK(cudaMalloc((void**)&d_output, sizeof(int) * TOP_K));
    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_input[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        // cudaMemset(d_output, 0, sizeof(int));
        topK<<<GRID_SIZE, BLOCK_SIZE>>>(d_output_1, d_input, n);

        topK<<<1, BLOCK_SIZE>>>(d_output, d_output_1, TOP_K * GRID_SIZE);

        CUDA_OK(cudaDeviceSynchronize());
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int) * TOP_K, cudaMemcpyDeviceToHost));

    // Verify result
    int cpu_topk[TOP_K] = {0};
    for (int i = 0; i < n; ++i) {
        insertTopK(cpu_topk, h_input[i]);
    }
    for (int i = 0; i < TOP_K; ++i) {
        printf("GPU topk[%d]: %d, CPU topk[%d]: %d\n", i, h_output[i], i, cpu_topk[i]);
        assert(h_output[i] == cpu_topk[i]);
    }

    printf("Top K completed successfully.\n");

    free(h_input);
    free(h_output);
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_output_1));
    CUDA_OK(cudaFree(d_output));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));;

    return 0;
}
