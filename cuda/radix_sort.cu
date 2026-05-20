#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <random>
#include <vector>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

#define BLOCK_SIZE 256

// Mark each element as 1 if the current radix bit is 0, otherwise 0.
__global__ void compute_zero_flags(unsigned int* zero_flags,
                                   const unsigned int* input,
                                   size_t n,
                                   int bit) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int bit_value = (input[i] >> bit) & 1u;
    zero_flags[i] = bit_value == 0u ? 1u : 0u;
}

// Compute block-local exclusive scan and emit one sum per block.
__global__ void exclusive_scan_blocks(unsigned int* output,
                                      unsigned int* block_sums,
                                      const unsigned int* input,
                                      size_t n) {
    __shared__ unsigned int shmem[BLOCK_SIZE];

    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    unsigned int x = (gid < n) ? input[gid] : 0u;
    shmem[threadIdx.x] = x;
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int add = 0u;
        if (threadIdx.x >= stride) {
            add = shmem[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            shmem[threadIdx.x] += add;
        }
        __syncthreads();
    }

    if (gid < n) {
        output[gid] = shmem[threadIdx.x] - x;
    }

    if (threadIdx.x == blockDim.x - 1) {
        block_sums[blockIdx.x] = shmem[threadIdx.x];
    }
}

// Add scanned block offsets to each element so local scans become global.
__global__ void add_block_offsets(unsigned int* output,
                                  const unsigned int* block_offsets,
                                  size_t n) {
    size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (gid >= n || blockIdx.x == 0) {
        return;
    }

    output[gid] += block_offsets[blockIdx.x];
}

// Hierarchical exclusive scan:
// 1) scan inside each block
// 2) recursively scan block sums
// 3) add scanned block offsets back
void exclusive_scan_u32(unsigned int* d_output, const unsigned int* d_input, size_t n) {
    if (n == 0) {
        return;
    }

    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto d_block_sums_buf = CudaBuffer<unsigned int>::with_capacity(num_blocks);
    unsigned int* d_block_sums = d_block_sums_buf.data();

    exclusive_scan_blocks<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_sums, d_input, n);
    CUDA_OK(cudaGetLastError());

    if (num_blocks > 1) {
        auto d_block_offsets_buf = CudaBuffer<unsigned int>::with_capacity(num_blocks);
        unsigned int* d_block_offsets = d_block_offsets_buf.data();

        exclusive_scan_u32(d_block_offsets, d_block_sums, num_blocks);

        add_block_offsets<<<num_blocks, BLOCK_SIZE>>>(d_output, d_block_offsets, n);
        CUDA_OK(cudaGetLastError());
    }
}

// Stable scatter by current bit:
// zeros go to [0, total_zeros), ones go after that range.
__global__ void scatter_by_bit(unsigned int* output,
                               const unsigned int* input,
                               const unsigned int* zero_flags,
                               const unsigned int* zero_positions,
                               size_t n,
                               unsigned int total_zeros) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int pos = 0;
    if (zero_flags[i] == 1u) {
        pos = zero_positions[i];
    } else {
        unsigned int ones_before = static_cast<unsigned int>(i) - zero_positions[i];
        pos = total_zeros + ones_before;
    }

    output[pos] = input[i];
}

// LSD radix sort for 32-bit unsigned integers.
// Each pass performs stable partition by one bit.
void radix_sort_gpu(unsigned int* d_output, const unsigned int* d_input, size_t n) {
    if (n == 0) {
        return;
    }

    auto d_ping_buf = CudaBuffer<unsigned int>::with_capacity(n);
    auto d_pong_buf = CudaBuffer<unsigned int>::with_capacity(n);
    auto d_zero_flags_buf = CudaBuffer<unsigned int>::with_capacity(n);
    auto d_zero_positions_buf = CudaBuffer<unsigned int>::with_capacity(n);

    unsigned int* d_ping = d_ping_buf.data();
    unsigned int* d_pong = d_pong_buf.data();
    unsigned int* d_zero_flags = d_zero_flags_buf.data();
    unsigned int* d_zero_positions = d_zero_positions_buf.data();

    CUDA_OK(cudaMemcpy(d_ping, d_input, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int bit = 0; bit < 32; ++bit) {
        // Step 1: classify elements by current bit.
        compute_zero_flags<<<num_blocks, BLOCK_SIZE>>>(d_zero_flags, d_ping, n, bit);
        CUDA_OK(cudaGetLastError());

        // Step 2: exclusive scan of zero flags gives write positions for zeros.
        exclusive_scan_u32(d_zero_positions, d_zero_flags, n);

        // Step 3: total number of zeros determines where ones start.
        unsigned int last_flag = 0;
        unsigned int last_pos = 0;
        CUDA_OK(cudaMemcpy(&last_flag, d_zero_flags + (n - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(&last_pos, d_zero_positions + (n - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost));
        unsigned int total_zeros = last_pos + last_flag;

        // Step 4: stable scatter into output buffer, then swap ping-pong buffers.
        scatter_by_bit<<<num_blocks, BLOCK_SIZE>>>(d_pong, d_ping, d_zero_flags, d_zero_positions, n, total_zeros);
        CUDA_OK(cudaGetLastError());

        std::swap(d_ping, d_pong);
    }

    CUDA_OK(cudaMemcpy(d_output, d_ping, n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}

bool verify_sorted(const unsigned int* data, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (data[i - 1] > data[i]) {
            fprintf(stderr,
                    "Array is not sorted at index %zu: %u > %u\n",
                    i,
                    data[i - 1],
                    data[i]);
            return false;
        }
    }
    return true;
}

bool verify_equal(const unsigned int* result, const unsigned int* expected, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (result[i] != expected[i]) {
            fprintf(stderr,
                    "Mismatch at index %zu: got %u, expected %u\n",
                    i,
                    result[i],
                    expected[i]);
            return false;
        }
    }
    return true;
}

void run_case(const std::vector<unsigned int>& input, const char* name) {
    size_t n = input.size();

    auto h_input_buf = HostBuffer<unsigned int>::with_capacity(n);
    auto h_output_buf = HostBuffer<unsigned int>::with_capacity(n);
    auto h_expected_buf = HostBuffer<unsigned int>::with_capacity(n);

    unsigned int* h_input = h_input_buf.data();
    unsigned int* h_output = h_output_buf.data();
    unsigned int* h_expected = h_expected_buf.data();

    for (size_t i = 0; i < n; ++i) {
        h_input[i] = input[i];
        h_expected[i] = input[i];
    }

    std::sort(h_expected, h_expected + n);

    auto d_input_buf = CudaBuffer<unsigned int>::with_capacity(n);
    auto d_output_buf = CudaBuffer<unsigned int>::with_capacity(n);
    unsigned int* d_input = d_input_buf.data();
    unsigned int* d_output = d_output_buf.data();

    d_input_buf.copy_from_host(h_input, n);

    CudaTimer timer;
    timer.start();
    radix_sort_gpu(d_output, d_input, n);
    float elapsed_ms = timer.elapsed_ms();

    d_output_buf.copy_to_host(h_output, n);
    d_output_buf.synchronize();

    assert(verify_sorted(h_output, n));
    assert(verify_equal(h_output, h_expected, n));

    printf("[%s] n=%zu, time=%f ms\n", name, n, elapsed_ms);
}

int main() {
    run_case({170u, 45u, 75u, 90u, 2u, 802u, 24u, 66u}, "Example 1");
    run_case({1u, 4u, 1u, 3u, 555u, 1000u, 2u}, "Example 2");

    std::vector<unsigned int> random_input(1 << 16);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<unsigned int> dist(0u, 0xffffffffu);
    for (size_t i = 0; i < random_input.size(); ++i) {
        random_input[i] = dist(rng);
    }
    run_case(random_input, "Random");

    printf("Radix sort completed successfully.\n");
    return 0;
}
