#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

const int NUM_REPS = 100;
const int GRID_SIZE = 32;
const int BLOCK_SIZE = 256;
const int WARP_SIZE = 32;
const int COARSE_FACTOR = 4;

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

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
    const size_t size = n * sizeof(int);

    int *h_input = (int*)malloc(size);
    int *h_output = (int*)malloc(sizeof(int));

    int *d_input, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, size));
    CUDA_OK(cudaMalloc((void**)&d_output, size));

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1;//i % 10;
    }

    int* expected = (int*)malloc(sizeof(int));
    cpu_sum(expected, h_input, n);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- Sum v1 --------------
    const unsigned int numThreadsPerBlock = BLOCK_SIZE;
    const unsigned int numElementsPerBlock = 2 * numThreadsPerBlock;
    const unsigned int numBlocks = (n + numElementsPerBlock - 1) / numElementsPerBlock;

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
        cudaMemset(d_output, 0, sizeof(int));
        sumV1<<<numBlocks, numThreadsPerBlock>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[baseline] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(*h_output == *expected);

    // -------------- Sum v2 --------------
    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
        cudaMemset(d_output, 0, sizeof(int));
        sumV2<<<numBlocks, numThreadsPerBlock>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[v2] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(*h_output == *expected);

    // -------------- Sum v3 --------------
    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        cudaMemset(d_output, 0, sizeof(int));
        sumV3<<<numBlocks, numThreadsPerBlock>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[shared memory] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(*h_output == *expected);

    // -------------- Sum v4 --------------
    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(startEvent));

    const unsigned int numThreadsPerBlock1 = BLOCK_SIZE;
    const unsigned int numElementsPerBlock1 = 2 * numThreadsPerBlock1;
    const unsigned int numBlocks1 = (n + numElementsPerBlock1 - 1) / numElementsPerBlock1;

    for (int i = 0; i < NUM_REPS; i++) {
        cudaMemset(d_output, 0, sizeof(int));
        sumV4<<<numBlocks1, numThreadsPerBlock1>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[thread coarsening] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(*h_output == *expected);

    // -------------- Sum v5 --------------
    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        cudaMemset(d_output, 0, sizeof(int));
        sumV5<<<GRID_SIZE, BLOCK_SIZE>>>(d_output, d_input, n);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[common] Average time per reduction: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(*h_output == *expected);

    // -------------- Sum using warp-shuffle kernel --------------

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
    assert(*h_output == *expected);

    printf("Reduction completed successfully.\n");

    free(h_input);
    free(h_output);
    free(expected);
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_output));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));;

    return 0;
}
