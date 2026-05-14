#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <vector>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

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

    auto input_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* input = input_buf.data();
    for (int i = 0; i < n; i++) {
        input[i] = i;
    }

    auto input_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto queue_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto queue_size_d_buf = CudaBuffer<int>::with_capacity(1);
    int* input_d = input_d_buf.data();
    int* queue_d = queue_d_buf.data();
    int* queue_size_d = queue_size_d_buf.data();

    input_d_buf.copy_from_host(input, static_cast<std::size_t>(n));

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* expected = expected_buf.data();
    int q_size = cpu_enqueue(expected, input, n);

    CudaTimer timer;
    float ms = 0.0f;

    unsigned int numThreadsPerBlock = 256;
    unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    //--------------- Wrap Vote V1 ----------------
    queue_size_d_buf.fill_zero();
    timer.start();

    enqueue_v1<<<numBlocks, numThreadsPerBlock>>>(queue_d, queue_size_d, input_d, n);

    ms = timer.elapsed_ms();
    printf("[v1] Time: %f ms\n", ms);

    // Copy output to CPU
    int queue_size = 0;
    queue_size_d_buf.copy_to_host(&queue_size, 1);
    queue_size_d_buf.synchronize();

    // Verify result
    assert(queue_size == q_size);

    //--------------- Wrap Vote V2 ----------------
    queue_size_d_buf.fill_zero();
    timer.start();

    enqueue_v2<<<numBlocks, numThreadsPerBlock>>>(queue_d, queue_size_d, input_d, n);

    ms = timer.elapsed_ms();
    printf("[v2] Time: %f ms\n", ms);

    // Copy output to CPU
    queue_size_d_buf.copy_to_host(&queue_size, 1);
    queue_size_d_buf.synchronize();

    // Verify result
    assert(queue_size == q_size);

    printf("Wrap Vote completed successfully.\n");
    return 0;
}
