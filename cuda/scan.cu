#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define COARSE_FACTOR 8

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

void cpu_scan(int* output, const int* input, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

__global__ void scan_kogge_stone(int* output, int* partial, const int* input, int n) {
    // Every thread has a corresponding element in the input and output array.
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: check bundary conditions.
    output[i] = input[i];
    // Make sure that this thread finishes writing to the output array before another thread reads from it.
    __syncthreads();

    // On every iteration, each thread adds the value that is stride to the left.
    // Not all the threads compute on every iteration.
    // For example, when stride is 1, all threads except the first one compute.
    // When stride is 2, all threads except the first two compute, and so on.
    // In general, for any value of stride, all threads with index >= stride do compute.
    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
        // Incorrect! Different threads are reading and writing the same data location without synchronization.
        // output[i] += output[i - stride];

        int val = 0;
        if (threadIdx.x >= stride) {
            val = output[i - stride];
        }
        // Put the sync threads outside of the boundary check.
        // Make sure all threads can actually reach the sync thread.
        __syncthreads();

        if (threadIdx.x >= stride) {
            output[i] += val;
        }
        __syncthreads();
    }

    // The last thread of each block writes its partial sum to the partial array.
    // TODO: check bundary conditions.
    if (threadIdx.x == blockDim.x - 1) {
        partial[blockIdx.x] = output[i];
    }
}

__global__ void scan_kogge_stone_sm(int* output, int* partial, const int* input, int n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int buffer_s[BLOCK_SIZE];
    // We are not going to use the global index of the thread to access the shared memory buffer.
    // We are going to use the local index of the thread to access the shared memory buffer.
    buffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
        int val = 0;
        if (threadIdx.x >= stride) {
            val = buffer_s[threadIdx.x - stride];
        }
        // Put the sync threads outside of the boundary check.
        // Make sure all threads can actually reach the sync thread.
        __syncthreads();

        if (threadIdx.x >= stride) {
            buffer_s[threadIdx.x] += val;
        }
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) {
        partial[blockIdx.x] = buffer_s[threadIdx.x];
    }

    // Write to the output array.
    output[i] = buffer_s[threadIdx.x];
}

__global__ void scan_kogge_stone_sm_db(int* output, int* partial, const int* input, int n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int buffer1_s[BLOCK_SIZE];
    __shared__ int buffer2_s[BLOCK_SIZE];

    // On each iteration, one buffer is used for reading and the other buffer is used for writing.
    int* inBuffer_s = buffer1_s;
    int* outputBuffer_s = buffer2_s;

    inBuffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
        if (threadIdx.x >= stride) {
            outputBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outputBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();

        // Swap buffers
        int* tmp = inBuffer_s;
        inBuffer_s = outputBuffer_s;
        outputBuffer_s = tmp;
    }

    if (threadIdx.x == blockDim.x - 1) {
        partial[blockIdx.x] = inBuffer_s[threadIdx.x];
    }

    output[i] = inBuffer_s[threadIdx.x];
}

__global__ void scan_kogge_stone_coarse(int* output, int* partial, const int* input, int n) { 
    size_t block_segment =  blockDim.x * COARSE_FACTOR * blockIdx.x;

    __shared__ int buffer_s[BLOCK_SIZE * COARSE_FACTOR];
    // We have COARSE_FACTOR sub-segments in each block, and each sub-segment has blockDim.x elements.
    for (size_t c = 0; c < COARSE_FACTOR; c++) {
        buffer_s[c * blockDim.x + threadIdx.x] = input[block_segment + c * blockDim.x + threadIdx.x];
    }
    __syncthreads();

    // Thread scan
    unsigned thread_segment = threadIdx.x * COARSE_FACTOR;
    for (size_t c = 1; c < COARSE_FACTOR; c++) {
        // Sequentially add the elements in the thread segment.
        buffer_s[thread_segment + c] += buffer_s[thread_segment + c - 1];
    }
    __syncthreads();

    // Each thread puts the sum of all the elements inside of its segment into the buffer.
    // And the thread block will do the scan of the partial sums in parallel.

    __shared__ int buffer1_s[BLOCK_SIZE];
    __shared__ int buffer2_s[BLOCK_SIZE];

    // On each iteration, one buffer is used for reading and the other buffer is used for writing.
    int* inBuffer_s = buffer1_s;
    int* outputBuffer_s = buffer2_s;

    // Put the partial sum of each thread segment into the buffer for the block scan.
    inBuffer_s[threadIdx.x] = buffer_s[thread_segment + COARSE_FACTOR - 1];
    __syncthreads();

    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride <<= 1) {
        if (threadIdx.x >= stride) {
            outputBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outputBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();

        // Swap buffers
        int* tmp = inBuffer_s;
        inBuffer_s = outputBuffer_s;
        outputBuffer_s = tmp;
    }

    // Each thread is going to add the partial sum of the previous threads to its own threads.
    if (threadIdx.x > 0) {
        for (size_t c = 0; c < COARSE_FACTOR; c++) {
            buffer_s[thread_segment + c] += inBuffer_s[threadIdx.x - 1];
        }
    }

    if (threadIdx.x == blockDim.x - 1) {
        partial[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    
    __syncthreads();

    // Just like at the very begining of the kernel, every thread loaded multiple input elements.
    // Now every thread is going to have to store multiple output elements.
    for (size_t c = 0; c < COARSE_FACTOR; c++) {
        output[block_segment + c * blockDim.x + threadIdx.x] = buffer_s[c * blockDim.x + threadIdx.x];
    }
}

__global__ void scan_brent_kung_sm(int* output, int* partial, const int* input, int n) {
    // We have twice as many input elements as threads in the block.
    size_t segment = blockIdx.x * blockDim.x * 2;

    // Every thread loads two elements.
    // And to do this in a coalesced way, we need to make all threads load the first half of the segment,
    // and then the second half.
    // TODO: check bundary conditions.
    __shared__  int buffer_s[BLOCK_SIZE * 2];
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[threadIdx.x + blockDim.x] = input[segment + threadIdx.x + blockDim.x];
    __syncthreads();

    // Reduction step
    for (size_t stride = 1; stride <= blockDim.x; stride <<= 1) {
        // Each thread processes element (t+1)*2*s-1.
        size_t i = (threadIdx.x + 1) * stride * 2 - 1;
        // Each thread is going to add the element it's responsible for to the element that is stride to the left.
        if (i < blockDim.x * 2) {
            buffer_s[i] += buffer_s[i - stride];
        }
        __syncthreads();
    }

    // Post reduction step
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // Each thread processes element (t+1)*2*s-1.
        size_t i = (threadIdx.x + 1) * stride * 2 - 1;
        // Each thread is going to add the element it's responsible for to the element that is stride to the right.
        // And store the result in the element that is stride to the right.
        if (i + stride < blockDim.x * 2) {
            buffer_s[i + stride] += buffer_s[i];
        }
        __syncthreads();
    }

    // Any thread can write the result to the partial array.
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = buffer_s[blockDim.x * 2 - 1];
    }

    // Just like each thread is responsible for two elements in the input array,
    // each thread is also responsible for writing two elements to the output array.
    // TODO: check bundary conditions.
    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + threadIdx.x + blockDim.x] = buffer_s[threadIdx.x + blockDim.x];
}

__global__ void add_partial(int* output, const int* partial, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        output[i] += partial[blockIdx.x - 1];
    }
}

__global__ void add_partial_v2(int* output, const int* partial, int n) {
    size_t block_segment =  blockDim.x * COARSE_FACTOR * blockIdx.x;

    if (blockIdx.x > 0) {
        for (size_t c = 0; c < COARSE_FACTOR; c++) {
            output[block_segment + c * blockDim.x + threadIdx.x] += partial[blockDim.x - 1];
        }
    }
}

void scan_d_kogge_stone(int* d_output, const int* d_input, int n, int kind) {
    const size_t numTreadsPerBlock = BLOCK_SIZE;
    const size_t numElementsPerBlock = numTreadsPerBlock;
    const size_t numBlocks = (n + numElementsPerBlock - 1) / numElementsPerBlock;

    int* d_partial;
    CUDA_OK(cudaMalloc((void**)&d_partial, numBlocks * sizeof(int)));

    if (kind == 0) {
        scan_kogge_stone<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, d_input, n);
    } else if (kind == 1) {
        scan_kogge_stone_sm<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, d_input, n);
    } else if (kind == 2) {
        scan_kogge_stone_sm_db<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, d_input, n);
    } else {
        fprintf(stderr, "Invalid kind: %d\n", kind);
        exit(1);
    }

    if (numBlocks > 1) {
        // Recursively calling the kernel again to do the scan for the partial array.
        scan_d_kogge_stone(d_partial, d_partial, numBlocks, kind);

        // After the scan for the partial array is done, we need to add the partial sums to the output array.
        add_partial<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, n);
    }

    cudaFree(d_partial);
    cudaDeviceSynchronize();
}

void scan_d_kogge_stone_v2(int* d_output, const int* d_input, int n) {
    const size_t numTreadsPerBlock = BLOCK_SIZE;
    const size_t numElementsPerBlock = numTreadsPerBlock * COARSE_FACTOR;
    const size_t numBlocks = (n + numElementsPerBlock - 1) / numElementsPerBlock;

    int* d_partial;
    CUDA_OK(cudaMalloc((void**)&d_partial, numBlocks * sizeof(int)));

    scan_kogge_stone_coarse<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, d_input, n);

    if (numBlocks > 1) {
        // Recursively calling the kernel again to do the scan for the partial array.
        scan_d_kogge_stone_v2(d_partial, d_partial, numBlocks);

        // After the scan for the partial array is done, we need to add the partial sums to the output array.
        add_partial_v2<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, n);
    }

    cudaFree(d_partial);
    cudaDeviceSynchronize();
}

void scan_d_brent_kung(int* d_output, const int* d_input, int n) {
    const size_t numTreadsPerBlock = BLOCK_SIZE;
    const size_t numElementsPerBlock = 2 * numTreadsPerBlock;
    const size_t numBlocks = (n + numElementsPerBlock - 1) / numElementsPerBlock;

    int* d_partial;
    CUDA_OK(cudaMalloc((void**)&d_partial, numBlocks * sizeof(int)));

    scan_brent_kung_sm<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, d_input, n);

    if (numBlocks > 1) {
        scan_d_brent_kung(d_partial, d_partial, numBlocks);
        add_partial<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, n);
    }

    cudaFree(d_partial);
    cudaDeviceSynchronize();
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
    int n = 1 << 16;

    int* h = (int*)malloc(n * sizeof(int));
    int* h_r = (int*)malloc(n * sizeof(int));

    int* d_input, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_input, n * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_output, n * sizeof(int)));\

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_input, h, n * sizeof(int), cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- Scan using Kogge-Stone algorithm --------------

    CUDA_OK(cudaEventRecord(startEvent));

    scan_d_kogge_stone(d_output, d_input, n, 0);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Scan using Kogge-Stone algorithm: %f ms\n", ms);

    CUDA_OK(cudaMemcpy(h_r, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    int* h_t = (int*)malloc(n * sizeof(int));
    cpu_scan(h_t, h, n);
    assert(verify_result(h_r, h_t, n));

    // -------------- Scan using Kogge-Stone algorithm with shared memory --------------

    CUDA_OK(cudaEventRecord(startEvent));

    scan_d_kogge_stone(d_output, d_input, n, 1);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Scan using Kogge-Stone algorithm with shared memory: %f ms\n", ms);

    CUDA_OK(cudaMemcpy(h_r, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    assert(verify_result(h_r, h_t, n));

    // -------------- Scan using Kogge-Stone algorithm with double buffering --------------

    CUDA_OK(cudaEventRecord(startEvent));

    scan_d_kogge_stone(d_output, d_input, n, 2);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Scan using Kogge-Stone algorithm with double buffering: %f ms\n", ms);

    CUDA_OK(cudaMemcpy(h_r, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    assert(verify_result(h_r, h_t, n));

    // -------------- Scan using Kogge-Stone algorithm with thread coarsening --------------

    CUDA_OK(cudaEventRecord(startEvent));

    scan_d_kogge_stone_v2(d_output, d_input, n);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Scan using Kogge-Stone algorithm with thread coarsening: %f ms\n", ms);

    CUDA_OK(cudaMemcpy(h_r, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    assert(verify_result(h_r, h_t, n));

    // -------------- Scan using brent kung algorithm with shared memory and double buffering --------------

    CUDA_OK(cudaEventRecord(startEvent));

    scan_d_brent_kung(d_output, d_input, n);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Scan using Brent-Kung algorithm with shared memory: %f ms\n", ms);

    CUDA_OK(cudaMemcpy(h_r, d_output, n * sizeof(int), cudaMemcpyDeviceToHost));

    assert(verify_result(h_r, h_t, n));

    printf("Scan completed successfully.\n");

    free(h);
    free(h_t);
    CUDA_OK(cudaFree(d_input));
    CUDA_OK(cudaFree(d_output));

    return 0;
}
