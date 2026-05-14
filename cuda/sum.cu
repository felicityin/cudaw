#include <stdio.h>
#include <assert.h>
#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int NUM_REPS = 100;
const int GRID_SIZE = 32;
const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;
const int COARSE_FACTOR = 4;

void cpu_sum(int* output, const int* input, int count) {
    *output = 0;
    for (int i = 0; i < count; i++) {
        *output += input[i];
    }
}

__global__ void sumV1(int *output, int *input, const int count) {
    // Every thread block is responsible for twice as many elements as it has threads.
    unsigned int segment = blockIdx.x * blockDim.x * 2;

    // Threads are distributed to every other element in the segment.
    int i = segment + threadIdx.x * 2;

    // In the first iteration, each thread adds its element with the next one.
    // In the second iteration, it adds the result with the next pair's result, and so on.
    for (int stride = 1; stride <= blockDim.x; stride <<= 1) {
        if (threadIdx.x % stride == 0 && i + stride < count) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    // The thread zero of each block will have a partial sum of the block's segment.
    // Add the sum of all blocks together.
    if (threadIdx.x == 0) {
        atomicAdd(output, input[segment]);
    }
}

__global__ void sumV2(int *output, int *input, const int count) {
    // Every thread block is responsible for twice as many elements as it has threads.
    unsigned int segment = blockIdx.x * blockDim.x * 2;

    unsigned int i = segment + threadIdx.x;

    for (int stride = blockDim.x; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && i + stride < count) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    // The thread zero of each block will have a partial sum of the block's segment.
    // Add the sum of all blocks together.
    if (threadIdx.x == 0) {
        atomicAdd(output, input[segment]);
    }
}

__global__ void sumV3(int *output, const int *input, const int count) {
    // Every thread block is responsible for twice as many elements as it has threads.
    unsigned int segment = blockIdx.x * blockDim.x * 2;

    unsigned int i = segment + threadIdx.x;

    __shared__ int input_s[BLOCK_SIZE];
    if (i + blockDim.x < count) {
        input_s[threadIdx.x] = input[i] + input[i + blockDim.x];
    } else if (i < count) {
        input_s[threadIdx.x] = input[i];
    } else {
        input_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // The thread zero of each block will have a partial sum of the block's segment.
    // Add the sum of all blocks together.
    if (threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}

__global__ void sumV4(int *output, const int *input, const int count) {
    unsigned int segment = blockIdx.x * blockDim.x * 2 * COARSE_FACTOR;

    unsigned int i = segment + threadIdx.x;

    __shared__ int input_s[BLOCK_SIZE];
    int sum = 0;
    for (int tile = 0; tile < COARSE_FACTOR * 2; tile++) {
        if (i + tile * blockDim.x < count) {
            sum += input[i + tile * blockDim.x];
        }
    }
    input_s[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // The thread zero of each block will have a partial sum of the block's segment.
    // Add the sum of all blocks together.
    if (threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}

__global__ void sumV5(int *output, const int *input, const int count) {
    __shared__ int s_mem[BLOCK_SIZE];

    // Grid stride loop to load data
    s_mem[threadIdx.x] = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        s_mem[threadIdx.x] += input[i];
    }

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (threadIdx.x < stride) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + stride];
        }
    }

    // The thread zero of each block will have a partial sum of the block's segment.
    // Add the sum of all blocks together.
    if (threadIdx.x == 0) {
        atomicAdd(output, s_mem[0]);
    }
}

__global__ void sumV6(int *output, const int *input, const int count) {
    __shared__ int s_mem[BLOCK_SIZE];

    // Grid stride loop to load data
    s_mem[threadIdx.x] = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        s_mem[threadIdx.x] += input[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > WARP_SIZE; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_mem[threadIdx.x] += s_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Warp reduction
    int sum = 0;
    if (threadIdx.x < WARP_SIZE) {
        // stride = WARP_SIZE
        sum = s_mem[threadIdx.x] + s_mem[threadIdx.x + WARP_SIZE];
    }
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        // Shuffle from a thread that has a higher index
        // The thread will take the sum value of the thread that's stride away from it,
        // and add it to its own sum value
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }

    // The thread zero of each block will have a partial sum of the block's segment.
    // Add the sum of all blocks together.
    if (threadIdx.x == 0) {
        atomicAdd(output, sum);
    }
}

__global__ void sumWarp(int *output, const int *input, const int count) {
    // Grid stride loop to load data
    int val = 0;
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        val += input[i];
    }

    // First warp-shuffle reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    __shared__ int s_mem[32];
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // Put warp results into shared memory
    if (laneId == 0) {
        s_mem[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0) {
        // Reload val from shared memory if warp existed
        int warpCount = blockDim.x / WARP_SIZE;
        val = (laneId < warpCount) ? s_mem[laneId] : 0;

        // Final warp-shuffle reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (laneId == 0) {    
            atomicAdd(output, val);
        }
    }
}

int main() {
    const int n = 1 << 10;

    auto h_input = HostBuffer<int>::with_capacity(n);
    h_input.set_len(n);
    auto h_output = HostBuffer<int>::with_capacity(1);
    h_output.set_len(1);

    auto d_input = CudaBuffer<int>::with_capacity(n);
    d_input.set_len(n);
    auto d_output = CudaBuffer<int>::with_capacity(1);
    d_output.set_len(1);

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1;  // i % 10;
    }

    auto expected = HostBuffer<int>::with_capacity(1);
    expected.set_len(1);
    cpu_sum(expected.data(), h_input.data(), n);

    // timer for timing
    CudaTimer timer;

    // -------------- Sum v1 --------------
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (n + numElementsPerBlock - 1) / numElementsPerBlock;

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        d_input.copy_from_host(h_input.data(), n);
        d_output.reset();
        sumV1<<<numBlocks, numThreadsPerBlock>>>(d_output.data(), d_input.data(), n);
    }

    float ms = timer.elapsed_ms();
    printf("[baseline] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    // -------------- Sum v2 --------------
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        d_input.copy_from_host(h_input.data(), n);
        d_output.reset();
        sumV2<<<numBlocks, numThreadsPerBlock>>>(d_output.data(), d_input.data(), n);
    }

    ms = timer.elapsed_ms();
    printf("[v2] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    // -------------- Sum v3 --------------
    d_input.copy_from_host(h_input.data(), n);
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        d_output.reset();
        sumV3<<<numBlocks, numThreadsPerBlock>>>(d_output.data(), d_input.data(), n);
    }

    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    // -------------- Sum v4 --------------
    d_input.copy_from_host(h_input.data(), n);
    timer.start();

    const unsigned int numThreadsPerBlock1 = BLOCK_SIZE;
    const unsigned int numElementsPerBlock1 = 2 * numThreadsPerBlock1;
    const unsigned int numBlocks1 = (n + numElementsPerBlock1 - 1) / numElementsPerBlock1;

    for (int i = 0; i < NUM_REPS; i++) {
        d_output.reset();
        sumV4<<<numBlocks1, numThreadsPerBlock1>>>(d_output.data(), d_input.data(), n);
    }

    ms = timer.elapsed_ms();
    printf("[thread coarsening] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    // -------------- Sum v5 --------------
    d_input.copy_from_host(h_input.data(), n);
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        d_output.reset();
        sumV5<<<GRID_SIZE, BLOCK_SIZE>>>(d_output.data(), d_input.data(), n);
    }

    ms = timer.elapsed_ms();
    printf("[v5] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    // -------------- Sum v6 --------------
    d_input.copy_from_host(h_input.data(), n);
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        d_output.reset();
        sumV6<<<GRID_SIZE, BLOCK_SIZE>>>(d_output.data(), d_input.data(), n);
    }

    ms = timer.elapsed_ms();
    printf("[v6] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    // -------------- Sum using warp-shuffle kernel --------------

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        d_output.reset();
        sumWarp<<<GRID_SIZE, BLOCK_SIZE>>>(d_output.data(), d_input.data(), n);
    }

    ms = timer.elapsed_ms();
    printf("[warp-shuffle] Average time per reduction: %f ms\n", ms / NUM_REPS);

    d_output.copy_to_host(h_output.data(), 1);

    // Verify result
    assert(h_output[0] == expected[0]);

    printf("Reduction completed successfully.\n");

    return 0;
}
