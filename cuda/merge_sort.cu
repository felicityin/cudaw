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
#define ITEMS_PER_THREAD 4

// Return i such that i elements are taken from A and k-i from B
// for the first k elements of merged(A, B).
__device__ __forceinline__ size_t co_rank(size_t k,
                                          const float* a,
                                          size_t m,
                                          const float* b,
                                          size_t n) {
    size_t i_low = (k > n) ? (k - n) : 0;
    size_t i_high = (k < m) ? k : m;

    while (i_low < i_high) {
        size_t i = i_low + (i_high - i_low) / 2;
        size_t j = k - i;

        if (i < m && j > 0 && b[j - 1] > a[i]) {
            i_low = i + 1;
        } else if (i > 0 && j < n && a[i - 1] > b[j]) {
            i_high = i;
        } else {
            return i;
        }
    }
    return i_low;
}

// One block merges one pair of sorted runs [left, mid) and [mid, right).
// Each thread owns a small output segment, and co_rank maps its boundaries
// to corresponding ranges in the two input runs.
__global__ void merge_pass_kernel(const float* src,
                                  float* dst,
                                  size_t n,
                                  size_t width) {
	// It indicates which "pair to be merged" the current block corresponds to.
	// Since the outer kernel launch is <<<num_pairs, BLOCK_SIZE>>>,
	// one block is responsible for one pair of sorted segments
    size_t pair_idx = static_cast<size_t>(blockIdx.x);
	// The total span of each pair of segments is 2 * width,
	// so the starting point of the pair with index pair_idx is pair_idx * 2 * width.
    size_t left = pair_idx * (width << 1);
    if (left >= n) {
        return;
    }

    size_t mid = left + width;
    size_t right = left + (width << 1);
    if (mid > n) {
        mid = n;
    }
    if (right > n) {
        right = n;
    }

    const float* a = src + left;
    const float* b = src + mid;
    size_t m = mid - left;
    size_t nn = right - mid;
    size_t total = m + nn;

	// If the total length of this pair of sorted segments exceeds `chunk`,
	// a single block will not complete it all at once.
	// Instead, it processes it in multiple chunks iteratively.
    size_t chunk = static_cast<size_t>(BLOCK_SIZE) * ITEMS_PER_THREAD;
    size_t block_chunks = (total + chunk - 1) / chunk;

    for (size_t c = 0; c < block_chunks; ++c) {
        size_t base_k = c * chunk;

        size_t k_begin = base_k + static_cast<size_t>(threadIdx.x) * ITEMS_PER_THREAD;
        size_t k_end = k_begin + ITEMS_PER_THREAD;
        if (k_begin >= total) {
            continue;
        }
        if (k_end > total) {
            k_end = total;
        }

        size_t i_begin = co_rank(k_begin, a, m, b, nn);
        size_t j_begin = k_begin - i_begin;

        size_t i_end = co_rank(k_end, a, m, b, nn);
        size_t j_end = k_end - i_end;

        size_t i = i_begin;
        size_t j = j_begin;
        size_t out = left + k_begin;

        while (i < i_end && j < j_end) {
            if (a[i] <= b[j]) {
                dst[out++] = a[i++];
            } else {
                dst[out++] = b[j++];
            }
        }
        while (i < i_end) {
            dst[out++] = a[i++];
        }
        while (j < j_end) {
            dst[out++] = b[j++];
        }
    }
}

void merge_sort_gpu(float* d_data, size_t n) {
    if (n <= 1) {
        return;
    }

    auto d_tmp_buf = CudaBuffer<float>::with_capacity(n);
    float* d_tmp = d_tmp_buf.data();

    float* ping = d_data;
    float* pong = d_tmp;
    bool ping_is_input = true;

    for (size_t width = 1; width < n; width <<= 1) {
        size_t num_pairs = (n + (width << 1) - 1) / (width << 1);

        merge_pass_kernel<<<num_pairs, BLOCK_SIZE>>>(ping, pong, n, width);
        CUDA_OK(cudaGetLastError());

        std::swap(ping, pong);
        ping_is_input = !ping_is_input;
    }

    // Ensure output is written back to the input array.
    if (!ping_is_input) {
        CUDA_OK(cudaMemcpy(d_data, ping, n * sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

bool verify_sorted(const float* data, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        if (data[i - 1] > data[i]) {
            fprintf(stderr,
                    "Array is not sorted at index %zu: %f > %f\n",
                    i,
                    data[i - 1],
                    data[i]);
            return false;
        }
    }
    return true;
}

bool verify_equal(const float* result, const float* expected, size_t n, float eps = 1e-6f) {
    for (size_t i = 0; i < n; ++i) {
        float diff = result[i] - expected[i];
        if (diff < 0.0f) {
            diff = -diff;
        }
        if (diff > eps) {
            fprintf(stderr,
                    "Mismatch at index %zu: got %f, expected %f\n",
                    i,
                    result[i],
                    expected[i]);
            return false;
        }
    }
    return true;
}

void run_case(const std::vector<float>& input, const char* name) {
    size_t n = input.size();

    auto h_input_buf = HostBuffer<float>::with_capacity(n);
    auto h_output_buf = HostBuffer<float>::with_capacity(n);
    auto h_expected_buf = HostBuffer<float>::with_capacity(n);

    float* h_input = h_input_buf.data();
    float* h_output = h_output_buf.data();
    float* h_expected = h_expected_buf.data();

    for (size_t i = 0; i < n; ++i) {
        h_input[i] = input[i];
        h_expected[i] = input[i];
    }

    std::sort(h_expected, h_expected + n);

    auto d_data_buf = CudaBuffer<float>::with_capacity(n);
    float* d_data = d_data_buf.data();

    d_data_buf.copy_from_host(h_input, n);

    CudaTimer timer;
    timer.start();
    merge_sort_gpu(d_data, n);
    float elapsed_ms = timer.elapsed_ms();

    d_data_buf.copy_to_host(h_output, n);
    d_data_buf.synchronize();

    assert(verify_sorted(h_output, n));
    assert(verify_equal(h_output, h_expected, n));

    printf("[%s] n=%zu, time=%f ms\n", name, n, elapsed_ms);
}

int main() {
    run_case({5.0f, 2.0f, 8.0f, 1.0f, 9.0f, 4.0f}, "Example");
    run_case({3.3f, -1.0f, 7.7f, 7.7f, 0.0f, -5.0f, 2.2f}, "Mixed");

    std::vector<float> random_input(1 << 16);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-100000.0f, 100000.0f);
    for (size_t i = 0; i < random_input.size(); ++i) {
        random_input[i] = dist(rng);
    }
    run_case(random_input, "Random");

    printf("Merge sort completed successfully.\n");
    return 0;
}
