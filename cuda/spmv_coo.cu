#include <stdio.h>
#include <assert.h>
#include <cstring>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

int NUM_REPS = 100;

template<typename T>
struct COOMatrix {
    int* row_indices;
    int* col_indices;
    T* values;
    int num_none_zeros;
    int num_rows, num_cols;
};

void cpu_spmv_coo(int* out_vec, const COOMatrix<int> matrix, const int* in_vec) {
    for (int i = 0; i < matrix.num_none_zeros; i++) {
        int row = matrix.row_indices[i];
        int col = matrix.col_indices[i];
        int val = matrix.values[i];
        out_vec[row] += val * in_vec[col];
    }
}

__global__ void spmv_coo(int* out_vec, const COOMatrix<int> matrix, const int* in_vec) { 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < matrix.num_none_zeros) {
        int row = matrix.row_indices[i];
        int col = matrix.col_indices[i];
        int val = matrix.values[i];
        atomicAdd(&out_vec[row], val * in_vec[col]);
    }
}

bool verify_result(const int* out_vec, const int* expected, int len) {
    for (int i = 0; i < len; i++) {
        if (out_vec[i] != expected[i]) {
            fprintf(stderr, "Mismatch at index %d: expected %d, got %d\n", i, expected[i], out_vec[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int row = 1 << 6;
    int col = 1 << 6;

    COOMatrix<int> coo_matrix;
    coo_matrix.num_rows = row;
    coo_matrix.num_cols = col;
    coo_matrix.num_none_zeros = 1 << 2;
    auto values_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(coo_matrix.num_none_zeros));
    auto row_indices_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(coo_matrix.num_none_zeros));
    auto col_indices_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(coo_matrix.num_none_zeros));
    coo_matrix.values = values_buf.data();
    coo_matrix.row_indices = row_indices_buf.data();
    coo_matrix.col_indices = col_indices_buf.data();
    for (int i = 0; i < coo_matrix.num_none_zeros; i++) {
        coo_matrix.values[i] = i % 1000;
        coo_matrix.row_indices[i] = i % row;
        coo_matrix.col_indices[i] = i % col;

    }

    auto in_vec_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(col));
    int* in_vec = in_vec_buf.data();
    for (int i = 0; i < col; i++) {
        in_vec[i] = i % 1000;
    }

    auto out_vec_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(row));
    int* out_vec = out_vec_buf.data();

    COOMatrix<int> coo_matrix_d;
    coo_matrix_d.num_rows = coo_matrix.num_rows;
    coo_matrix_d.num_cols = coo_matrix.num_cols;
    coo_matrix_d.num_none_zeros = coo_matrix.num_none_zeros;
    auto values_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(coo_matrix.num_none_zeros));
    auto row_indices_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(coo_matrix.num_none_zeros));
    auto col_indices_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(coo_matrix.num_none_zeros));
    coo_matrix_d.values = values_d_buf.data();
    coo_matrix_d.row_indices = row_indices_d_buf.data();
    coo_matrix_d.col_indices = col_indices_d_buf.data();
    CUDA_OK(cudaMemcpy(coo_matrix_d.values, coo_matrix.values, coo_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(coo_matrix_d.row_indices, coo_matrix.row_indices, coo_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(coo_matrix_d.col_indices, coo_matrix.col_indices, coo_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));

    auto in_vec_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(col));
    int* in_vec_d = in_vec_d_buf.data();
    CUDA_OK(cudaMemcpy(in_vec_d, in_vec, col * sizeof(int), cudaMemcpyHostToDevice));

    auto out_vec_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(row));
    int* out_vec_d = out_vec_d_buf.data();

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(row));
    int* expected = expected_buf.data();
    memset(expected, 0, row * sizeof(int));
    cpu_spmv_coo(expected, coo_matrix, in_vec);

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- SpMV using COO --------------
    timer.start();
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (coo_matrix_d.num_none_zeros + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(out_vec_d, 0, row * sizeof(int)));
        spmv_coo<<<numBlocks, numThreadsPerBlock>>>(out_vec_d, coo_matrix_d, in_vec_d);
    }
    ms = timer.elapsed_ms();
    printf("[COO] Average time per SpMV: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(out_vec, out_vec_d, row * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(out_vec, expected, row));
    printf("SpMV COO completed successfully.\n");
    return 0;
}
