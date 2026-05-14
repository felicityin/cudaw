#include <stdio.h>
#include <assert.h>
#include <cstring>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int NUM_REPS = 100;
const int NUM_BINS = 256;
const int CORSE_FACTOR = 16;

void cpu_histogram(unsigned int* bins, const unsigned char* image, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        unsigned char pixel = image[i];
        bins[pixel]++;
    }
}

__global__ void histogram(unsigned int* bins, const unsigned char* image, int width, int height) {
    // Assign a thread to every pixel in the image.
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        unsigned char pixel = image[i];
        atomicAdd(&bins[pixel], 1);
    }
}

__global__ void histogram_mm(unsigned int* bins, const unsigned char* image, int width, int height) {
    __shared__ unsigned int bins_s[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        bins_s[i] = 0;
    }
    __syncthreads();

    // Assign a thread to every pixel in the image.
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < width * height) {
        unsigned char pixel = image[i];
        atomicAdd(&bins_s[pixel], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&bins[i], bins_s[i]);
    }
}

__global__ void histogram_mm_corse(unsigned int* bins, const unsigned char* image, int width, int height) {
    __shared__ unsigned int bins_s[NUM_BINS];

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        bins_s[i] = 0;
    }
    __syncthreads();

    unsigned int segment = blockIdx.x * blockDim.x * CORSE_FACTOR;

    // We have COARSE_FACTOR sub-segments in each block, and each sub-segment has blockDim.x elements.
    for (int c = 0; c < CORSE_FACTOR; c++) {
        int i = segment + c * blockDim.x + threadIdx.x;
         if (i < width * height) {
            unsigned char pixel = image[i];
            atomicAdd(&bins_s[pixel], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&bins[i], bins_s[i]);
    }
}

bool verify_result(const unsigned int* result, const unsigned int* expected, int n) {
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

    auto image_buf = HostBuffer<unsigned char>::with_capacity(static_cast<std::size_t>(n));
    auto bins_buf = HostBuffer<unsigned int>::with_capacity(NUM_BINS);
    auto expected_buf = HostBuffer<unsigned int>::with_capacity(NUM_BINS);

    unsigned char* image = image_buf.data();
    unsigned int* bins = bins_buf.data();
    unsigned int* expected = expected_buf.data();

    auto d_image_buf = CudaBuffer<unsigned char>::with_capacity(static_cast<std::size_t>(n));
    auto d_bins_buf = CudaBuffer<unsigned int>::with_capacity(NUM_BINS);
    unsigned char* d_image = d_image_buf.data();
    unsigned int* d_bins = d_bins_buf.data();

    // Initialize input
    for (int i = 0; i < n; ++i) {
        image[i] = i % NUM_BINS;
    }

    memset(expected, 0, NUM_BINS * sizeof(unsigned int));
    cpu_histogram(expected, image, width, height);

    CUDA_OK(cudaMemcpy(d_image, image, n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- Histogram using naive kernel --------------
    timer.start();
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(d_bins, 0, 256 * sizeof(int)));
        histogram<<<numBlocks, numThreadsPerBlock>>>(d_bins, d_image, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[baseline] Average time per histogram: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(bins, d_bins, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(bins, expected, 256));

    // -------------- Histogram using shared memory --------------
    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(d_bins, 0, 256 * sizeof(int)));
        histogram_mm<<<numBlocks, numThreadsPerBlock>>>(d_bins, d_image, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per histogram: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(bins, d_bins, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(bins, expected, 256));

    // -------------- Histogram using thread coarsening --------------
    timer.start();
    unsigned int numThreadsPerBlock1 = 1024;
    unsigned int numBlocks1 = (n + numThreadsPerBlock1 - 1) / numThreadsPerBlock1 / CORSE_FACTOR;

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(d_bins, 0, 256 * sizeof(int)));
        histogram_mm_corse<<<numBlocks1, numThreadsPerBlock1>>>(d_bins, d_image, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[thread coarsening] Average time per histogram: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(bins, d_bins, 256 * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(bins, expected, 256));

    printf("Histogram completed successfully.\n");

    return 0;
}
