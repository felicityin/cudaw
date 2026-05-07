#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

const int WARP_SIZE = 32;

__host__ __device__ bool cond(int x) {
    return x % 2 == 0;
}

int cpu_enqueue(int* queue, const int* input, int n) {
    int j = 0;
    for (int i = 0; i < n; i++) {
        int val = input[i];
        if (cond(val)) {
            queue[j++] = val;
        }
    }
    return j;
}

__global__ void enqueue_v1(int* queue, int* queue_size, const int* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = input[i];
        if (cond(val)) {
            int j = atomicAdd(queue_size, 1);
            queue[j] = val;
        }
    }
}

__global__ void enqueue_v2(int* queue, int* queue_size, const int* input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int val = input[i];
        if (cond(val)) {
            // Let each warp rather than every thread doing an atomic add.
            // We can have the threads of warp collaborate with each other to figure out
            // haw many threads of the warp want to add to the queue, and then have only
            // one thread do the atomic add.

            // Assign a leader thread
            int active_threads = __activemask();
            int leader = __ffs(active_threads) - 1;
            // Find how many threads are active
            int num_active_threads = __popc(active_threads);
            // Leader allocate in queue
            int j;
            if (threadIdx.x % WARP_SIZE == leader) {
                j = atomicAdd(queue_size, num_active_threads);
            }
            // Shuffle the leader's j to all other threads
            j = __shfl_sync(active_threads, j, leader);
            // Find the position of each active thread in the queue
            int prev_threads_mask = (1 << (threadIdx.x % WARP_SIZE)) - 1;
            int prev_active_threads = active_threads & prev_threads_mask;
            int offset = __popc(prev_active_threads);
            // Store the result
            queue[j + offset] = threadIdx.x;
        }
    }
}

int main() {
    int n = 1 << 10;

    int* input = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        input[i] = i;
    }

    int* input_d;
    int* queue_d;
    int* queue_size_d;
    CUDA_OK(cudaMalloc((void**)&input_d, n * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&queue_d, n * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&queue_size_d, sizeof(int)));

    CUDA_OK(cudaMemcpy(input_d, input, n * sizeof(int), cudaMemcpyHostToDevice));

    int* expected = (int*)malloc(n * sizeof(int));
    int q_size = cpu_enqueue(expected, input, n);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    unsigned int numThreadsPerBlock = 256;
    unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    //--------------- Wrap Vote V1 ----------------
    CUDA_OK(cudaMemset(queue_size_d, 0, sizeof(int)));
    CUDA_OK(cudaEventRecord(startEvent));

    enqueue_v1<<<numBlocks, numThreadsPerBlock>>>(queue_d, queue_size_d, input_d, n);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[v1] Time: %f ms\n", ms);

    // Copy output to CPU
    int queue_size = 0;
    CUDA_OK(cudaMemcpy(&queue_size, queue_size_d, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(queue_size == q_size);

    //--------------- Wrap Vote V2 ----------------
    CUDA_OK(cudaMemset(queue_size_d, 0, sizeof(int)));
    CUDA_OK(cudaEventRecord(startEvent));

    enqueue_v2<<<numBlocks, numThreadsPerBlock>>>(queue_d, queue_size_d, input_d, n);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[v2] Time: %f ms\n", ms);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(&queue_size, queue_size_d, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(queue_size == q_size);

    printf("Wrap Vote completed successfully.\n");

    free(expected);
    CUDA_OK(cudaFree(queue_d));
    CUDA_OK(cudaFree(queue_size_d));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));;
    return 0;
}
