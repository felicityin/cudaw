#include <stdio.h>
#include <assert.h>
#include <cstring>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

int NUM_REPS = 100;

template<typename T>
struct ELLMatrix {
    int num_rows, num_cols;
    int max_nnz_per_row;
    int* nnz_per_row;
    int* col_indices;
    T* values;
};

void cpu_spmv_ell(int* out_vec, const ELLMatrix<int>& matrix, const int* in_vec) {
    for (int row = 0; row < matrix.num_rows; row++) {
        int sum = 0;
        for (unsigned int iter = 0; iter < matrix.nnz_per_row[row]; ++iter) {
            unsigned int i = iter * matrix.num_rows + row;
            unsigned int col = matrix.col_indices[i];
            int val = matrix.values[i];
            sum += val * in_vec[col];
        }
        out_vec[row] = sum;
    }
}

__global__ void spmv_ell(int* out_vec, ELLMatrix<int> matrix, const int* in_vec) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrix.num_rows) {
        int sum = 0;
        for (unsigned int iter = 0; iter < matrix.nnz_per_row[row]; ++iter) {
            unsigned int i = iter * matrix.num_rows + row;
            unsigned int col = matrix.col_indices[i];
            int val = matrix.values[i];
            sum += val * in_vec[col];
        }
        out_vec[row] = sum;
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
    int rows = 1 << 6;
    int cols = 1 << 6;
    int nnz = 1 << 2;

    struct CSRMatrix {
        int* row_ptr;
        int* col_indices;
        int* values;
        int num_none_zeros;
        int num_rows, num_cols;
    };
    CSRMatrix csr_matrix;
    csr_matrix.num_rows = rows;
    csr_matrix.num_cols = cols;
    csr_matrix.num_none_zeros = nnz;
    auto csr_row_ptr_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix.num_rows + 1));
    auto csr_col_indices_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix.num_none_zeros));
    auto csr_values_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix.num_none_zeros));
    csr_matrix.row_ptr = csr_row_ptr_buf.data();
    csr_matrix.col_indices = csr_col_indices_buf.data();
    csr_matrix.values = csr_values_buf.data();
    csr_matrix.row_ptr[0] = 0;

    // Build a valid CSR row_ptr for any nnz/row ratio.
    int base_nnz_per_row = csr_matrix.num_none_zeros / csr_matrix.num_rows;
    int remainder = csr_matrix.num_none_zeros % csr_matrix.num_rows;
    int offset = 0;
    for (int r = 0; r < csr_matrix.num_rows; r++) {
        int row_nnz = base_nnz_per_row + (r < remainder ? 1 : 0);
        offset += row_nnz;
        csr_matrix.row_ptr[r + 1] = offset;
    }
    for (int i = 0; i < csr_matrix.num_none_zeros; i++) {
        csr_matrix.values[i] = i % 1000;
        csr_matrix.col_indices[i] = i % cols;
    }

    ELLMatrix<int> ell_matrix;
    ell_matrix.num_rows = rows;
    ell_matrix.num_cols = cols;
    auto nnz_per_row_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(rows));
    ell_matrix.nnz_per_row = nnz_per_row_buf.data();

    int max_nnz_per_row = 0;
    for (int i = 0; i < rows; ++i) {
        int nnz = csr_matrix.row_ptr[i + 1] - csr_matrix.row_ptr[i];
        ell_matrix.nnz_per_row[i] = nnz;
        max_nnz_per_row = max_nnz_per_row < nnz ? nnz : max_nnz_per_row;
    }
    ell_matrix.max_nnz_per_row = max_nnz_per_row;
    auto ell_col_indices_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(max_nnz_per_row) * rows);
    auto ell_values_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(max_nnz_per_row) * rows);
    ell_matrix.col_indices = ell_col_indices_buf.data();
    ell_matrix.values = ell_values_buf.data();

    for (int i = 0; i < rows; ++i) {
        int row_start = csr_matrix.row_ptr[i];
        int row_end = csr_matrix.row_ptr[i + 1];
        
        for (int k = 0; k < row_end - row_start; ++k) {
            int idx = i * max_nnz_per_row + k;
            int coo_idx = row_start + k;
            
            ell_matrix.col_indices[idx] = csr_matrix.col_indices[coo_idx];
            ell_matrix.values[idx] = csr_matrix.values[coo_idx];
        }
    }

    auto in_vec_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(cols));
    int* in_vec = in_vec_buf.data();
    for (int i = 0; i < cols; i++) {
        in_vec[i] = i % 1000;
    }

    auto out_vec_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(rows));
    int* out_vec = out_vec_buf.data();

    ELLMatrix<int> ell_matrix_d;
    ell_matrix_d.num_rows = ell_matrix.num_rows;
    ell_matrix_d.num_cols = ell_matrix.num_cols;
    ell_matrix_d.max_nnz_per_row = ell_matrix.max_nnz_per_row;
    auto nnz_per_row_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(rows));
    auto ell_col_indices_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(max_nnz_per_row) * rows);
    auto ell_values_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(max_nnz_per_row) * rows);
    ell_matrix_d.nnz_per_row = nnz_per_row_d_buf.data();
    ell_matrix_d.col_indices = ell_col_indices_d_buf.data();
    ell_matrix_d.values = ell_values_d_buf.data();
    nnz_per_row_d_buf.copy_from_host(ell_matrix.nnz_per_row, static_cast<std::size_t>(rows));
    ell_values_d_buf.copy_from_host(ell_matrix.values, static_cast<std::size_t>(max_nnz_per_row) * rows);
    ell_col_indices_d_buf.copy_from_host(ell_matrix.col_indices, static_cast<std::size_t>(max_nnz_per_row) * rows);

    auto in_vec_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(cols));
    int* in_vec_d = in_vec_d_buf.data();
    in_vec_d_buf.copy_from_host(in_vec, static_cast<std::size_t>(cols));

    auto out_vec_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(rows));
    int* out_vec_d = out_vec_d_buf.data();

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(rows));
    int* expected = expected_buf.data();
    memset(expected, 0, rows * sizeof(int));
    cpu_spmv_ell(expected, ell_matrix, in_vec);

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- SpMV using ELL --------------
    timer.start();
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (ell_matrix.num_rows + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
        out_vec_d_buf.fill_zero();
        spmv_ell<<<numBlocks, numThreadsPerBlock>>>(out_vec_d, ell_matrix_d, in_vec_d);
        CUDA_OK(cudaGetLastError());
    }
    ms = timer.elapsed_ms();
    printf("[ELL] Average time per SpMV: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    out_vec_d_buf.copy_to_host(out_vec, static_cast<std::size_t>(rows));
    out_vec_d_buf.synchronize();

    // Verify result
    assert(verify_result(out_vec, expected, rows));
    printf("SpMV ELL completed successfully.\n");
    return 0;
}
