#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024

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
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    output[i] = input[i];
    __syncthreads();

    for (unsigned int offset = 1; offset <= blockDim.x / 2; offset <<= 1) {
        int val = 0;
        if (threadIdx.x >= offset) {
            val = output[i - offset];
        }
        // Put the sync threads outside of the boundary check.
        // Make sure all threads can actually reach the sync thread.
        __syncthreads();
        if (threadIdx.x >= offset) {
            output[i] += val;
        }
        __syncthreads();
    }
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

    for (unsigned int offset = 1; offset <= blockDim.x / 2; offset <<= 1) {
        int val = 0;
        if (threadIdx.x >= offset) {
            val = buffer_s[threadIdx.x - offset];
        }
        // Put the sync threads outside of the boundary check.
        // Make sure all threads can actually reach the sync thread.
        __syncthreads();
        if (threadIdx.x >= offset) {
            buffer_s[threadIdx.x] += val;
        }
        __syncthreads();
    }
    if (threadIdx.x == blockDim.x - 1) {
        partial[blockIdx.x] = buffer_s[threadIdx.x];
    }
    output[i] = buffer_s[threadIdx.x];
}

__global__ void scan_kogge_stone_sm_db(int* output, int* partial, const int* input, int n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int buffer1_s[BLOCK_SIZE];
    __shared__ int buffer2_s[BLOCK_SIZE];
    int* inBuffer_s = buffer1_s;
    int* outputBuffer_s = buffer2_s;
    inBuffer_s[threadIdx.x] = input[i];
    __syncthreads();

    for (unsigned int offset = 1; offset <= blockDim.x / 2; offset <<= 1) {
        if (threadIdx.x >= offset) {
            outputBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - offset];
        } else {
            outputBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        __syncthreads();
        int* tmp = inBuffer_s;
        inBuffer_s = outputBuffer_s;
        outputBuffer_s = tmp;
    }
    if (threadIdx.x == blockDim.x - 1) {
        partial[blockIdx.x] = inBuffer_s[threadIdx.x];
    }
    output[i] = inBuffer_s[threadIdx.x];
}

__global__ void scan_brent_kung_sm(int* output, int* partial, const int* input, int n) {
    size_t segment = blockIdx.x * blockDim.x * 2;

    __shared__  int buffer_s[BLOCK_SIZE * 2];
    buffer_s[threadIdx.x] = input[segment + threadIdx.x];
    buffer_s[blockDim.x + threadIdx.x] = input[segment + blockDim.x + threadIdx.x];
    __syncthreads();

    // Reduction step
    for (size_t offset = 1; offset <= blockDim.x; offset <<= 1) {
        size_t i = (threadIdx.x + 1) * offset * 2 - 1;
        if (i < blockDim.x * 2) {
            buffer_s[i] += buffer_s[i - offset];
        }
        __syncthreads();
    }

    // Post reduction step
    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        size_t i = (threadIdx.x + 1) * offset * 2 - 1;
        if (i + offset < blockDim.x * 2) {
            buffer_s[i + offset] += buffer_s[i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = buffer_s[blockDim.x * 2 - 1];
    }

    output[segment + threadIdx.x] = buffer_s[threadIdx.x];
    output[segment + blockDim.x + threadIdx.x] = buffer_s[blockDim.x + threadIdx.x];
}

__global__ void add_partial(int* output, const int* partial, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0) {
        output[i] += partial[blockIdx.x - 1];
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
    } else if (kind == 3) {
        scan_brent_kung_sm<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, d_input, n);
    } else {
        fprintf(stderr, "Invalid kind: %d\n", kind);
        exit(1);
    }

    if (numBlocks > 1) {
        scan_d_kogge_stone(d_partial, d_partial, numBlocks, kind);
        add_partial<<<numBlocks, numElementsPerBlock>>>(d_output, d_partial, n);
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

    // -------------- Scan using Kogge-Stone algorithm with shared memory and double buffering --------------

    CUDA_OK(cudaEventRecord(startEvent));

    scan_d_kogge_stone(d_output, d_input, n, 2);

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("Scan using Kogge-Stone algorithm with shared memory and double buffering: %f ms\n", ms);

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
