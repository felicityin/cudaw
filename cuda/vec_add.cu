#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

__global__ void add(int *output, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < n){
        output[i] = a[i] + b[i];
    }
}

__global__ void addV2(int *output, const int *a, const int *b, int n) {
    for (
        int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < n; 
        i += blockDim.x * gridDim.x
    ) {
        output[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t size = n * sizeof(int);

    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_output = (int*)malloc(size);

    // Allocate GPU memory
    int *d_a, *d_b, *d_output;
    CUDA_OK(cudaMalloc((void**)&d_a, size));
    CUDA_OK(cudaMalloc((void**)&d_b, size));
    CUDA_OK(cudaMalloc((void**)&d_output, size));

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    int numSMs;
    CUDA_OK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    printf("num sms: %d\n", numSMs);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // ------------- v1 ------------------
    CUDA_OK(cudaEventRecord(startEvent));

    // Copy input to GPU
    CUDA_OK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Call a GPU kenrel function (launch a grid of threads)
    addV2<<<32 * numSMs, 256>>>(d_output, d_a, d_b, n);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[v1] time: %f ms\n", ms);

    // Verify result
    for (int i = 0; i < n; ++i) {
        assert(h_output[i] == h_a[i] + h_b[i]);
    }

    free(h_a);
    free(h_b);
    free(h_output);

    // ------------- pinned memory ------------------
    cudaMallocHost((void**)&h_a, size);
    cudaMallocHost((void**)&h_b, size);
    cudaMallocHost((void**)&h_output, size);

    CUDA_OK(cudaEventRecord(startEvent));

    // Copy input to GPU
    CUDA_OK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Call a GPU kenrel function (launch a grid of threads)
    addV2<<<32 * numSMs, 256>>>(d_output, d_a, d_b, n);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[pinned memory] time: %f ms\n", ms);

    // Verify result
    for (int i = 0; i < n; ++i) {
        assert(h_output[i] == h_a[i] + h_b[i]);
    }

    // ------------- stream ------------------
    CUDA_OK(cudaEventRecord(startEvent));

    // Setup streams
    const int num_streams = 32;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        CUDA_OK(cudaStreamCreate(&streams[i]));
    }

    // Stream the segments
    const int num_segments = num_streams;
    const int segment_size = (n + num_segments - 1) / num_segments;

    for (int i = 0; i < num_segments; ++i) {
        // Find the segment bounds
        int start = i * segment_size;
        int end = start + segment_size < n ? start + segment_size : n;
        int n_segment = end - start;

        // Copy input to GPU
        CUDA_OK(cudaMemcpyAsync(d_a + start, h_a + start, n_segment, cudaMemcpyHostToDevice, streams[i]));
        CUDA_OK(cudaMemcpyAsync(d_b + start, h_b + start, n_segment, cudaMemcpyHostToDevice, streams[i]));

        // Call a GPU kenrel function (launch a grid of threads)
        addV2<<<32 * numSMs, 256, 0, streams[i]>>>(d_output + start, d_a + start, d_b + start, n_segment);

        // Copy output to CPU
        CUDA_OK(cudaMemcpyAsync(h_output + start, d_output + start, n_segment, cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[stream] time: %f ms\n", ms);

    // Verify result
    for (int i = 0; i < n; ++i) {
        assert(h_output[i] == h_a[i] + h_b[i]);
    }

    printf("Add completed successfully.\n");

    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_output);
    // Deallocate GPU memory
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaFree(d_output));

    return 0;
}
