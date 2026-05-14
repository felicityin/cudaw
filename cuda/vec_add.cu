#include <stdio.h>
#include <assert.h>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/stream.cuh"
#include "include/timer.cuh"

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

    auto h_a_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto h_b_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto h_output_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* h_a = h_a_buf.data();
    int* h_b = h_b_buf.data();
    int* h_output = h_output_buf.data();

    auto d_a_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto d_b_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto d_output_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* d_a = d_a_buf.data();
    int* d_b = d_b_buf.data();
    int* d_output = d_output_buf.data();

    // Initialize input
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    int numSMs;
    CUDA_OK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    printf("num sms: %d\n", numSMs);

    CudaTimer timer;
    float ms = 0.0f;

    // ------------- v1 ------------------
    timer.start();

    // Copy input to GPU
	d_a_buf.copy_from_host(h_a, n);
	d_b_buf.copy_from_host(h_b, n);

    // Call a GPU kenrel function (launch a grid of threads)
    addV2<<<32 * numSMs, 256>>>(d_output, d_a, d_b, n);
    CUDA_OK(cudaGetLastError());

    // Copy output to CPU
    d_output_buf.copy_to_host(h_output, n);
    d_output_buf.synchronize();

    ms = timer.elapsed_ms();
    printf("[v1] time: %f ms\n", ms);

    // Verify result
    for (int i = 0; i < n; ++i) {
        assert(h_output[i] == h_a[i] + h_b[i]);
    }

    // ------------- pinned memory ------------------
    auto pin_a = PinBuffer<int>::with_capacity(n);
    auto pin_b = PinBuffer<int>::with_capacity(n);
    auto pin_output = PinBuffer<int>::with_capacity(n);

    h_a = pin_a.data();
    h_b = pin_b.data();
    h_output = pin_output.data();

    // Initialize input (pinned)
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    timer.start();

    // Copy input to GPU
	d_a_buf.copy_from_host(h_a, n);
	d_b_buf.copy_from_host(h_b, n);

    // Call a GPU kenrel function (launch a grid of threads)
    addV2<<<32 * numSMs, 256>>>(d_output, d_a, d_b, n);
    CUDA_OK(cudaGetLastError());

    // Copy output to CPU
	d_output_buf.copy_to_host(h_output, n);
    d_output_buf.synchronize();

    ms = timer.elapsed_ms();
    printf("[pinned memory] time: %f ms\n", ms);

    // Verify result
    for (int i = 0; i < n; ++i) {
    assert(h_output[i] == h_a[i] + h_b[i]);
    }

    // ------------- stream ------------------
    timer.start();

    // Setup streams
    const int num_streams = 32;
    CudaStream streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        streams[i] = CudaStream::create();
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
		streams[i].memcpy_host_to_device_async(d_a + start, h_a + start, n_segment);
		streams[i].memcpy_host_to_device_async(d_b + start, h_b + start, n_segment);

        // Call a GPU kenrel function (launch a grid of threads)
        addV2<<<32 * numSMs, 256, 0, streams[i].get()>>>(
            d_output + start, d_a + start, d_b + start, n_segment);
        CUDA_OK(cudaGetLastError());

        // Copy output to CPU
		streams[i].memcpy_device_to_host_async(h_output + start, d_output + start, n_segment);
    }

    for (int i = 0; i < num_streams; ++i) {
        streams[i].synchronize();
    }
    ms = timer.elapsed_ms();
    printf("[stream] time: %f ms\n", ms);

    // Verify result
    for (int i = 0; i < n; ++i) {
        assert(h_output[i] == h_a[i] + h_b[i]);
    }

    printf("Add completed successfully.\n");

    return 0;
}
