#include <stdio.h>
#include <assert.h>
#include <cstring>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

int NUM_REPS = 100;

template<typename T>
struct CSRMatrix {
    int* row_ptr;
    int* col_indices;
    T* values;
    int num_none_zeros;
    int num_rows, num_cols;
};

bool validate_csr(const CSRMatrix<int>& matrix) {
    if (matrix.row_ptr[0] != 0) {
        fprintf(stderr, "Invalid CSR: row_ptr[0] must be 0\n");
        return false;
    }
    for (int r = 0; r < matrix.num_rows; r++) {
        if (matrix.row_ptr[r] > matrix.row_ptr[r + 1]) {
            fprintf(stderr, "Invalid CSR: row_ptr is not non-decreasing at row %d\n", r);
            return false;
        }
    }
    if (matrix.row_ptr[matrix.num_rows] != matrix.num_none_zeros) {
        fprintf(stderr, "Invalid CSR: row_ptr[num_rows]=%d but nnz=%d\n",
                matrix.row_ptr[matrix.num_rows], matrix.num_none_zeros);
        return false;
    }
    for (int i = 0; i < matrix.num_none_zeros; i++) {
        if (matrix.col_indices[i] < 0 || matrix.col_indices[i] >= matrix.num_cols) {
            fprintf(stderr, "Invalid CSR: col_indices[%d]=%d out of range [0, %d)\n",
                    i, matrix.col_indices[i], matrix.num_cols);
            return false;
        }
    }
    return true;
}

void cpu_spmv_csr(int* out_vec, const CSRMatrix<int>& matrix, const int* in_vec) {
    for (int row = 0; row < matrix.num_rows; row++) {
        int sum = 0;
        for (int i = matrix.row_ptr[row]; i < matrix.row_ptr[row + 1]; i++) {
            int col = matrix.col_indices[i];
            int val = matrix.values[i];
            sum += val * in_vec[col];
        }
        out_vec[row] = sum;
    }
}

__global__ void spmv_csr(int* out_vec, CSRMatrix<int> matrix, const int* in_vec) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrix.num_rows) {
        int row_start = matrix.row_ptr[row];
        int row_end = matrix.row_ptr[row + 1];
        if (row_start < 0 || row_end < row_start || row_end > matrix.num_none_zeros) {
            out_vec[row] = 0;
            return;
        }

        int sum = 0;
        for (int i = row_start; i < row_end; i++) {
            int col = matrix.col_indices[i];
            if (col < 0 || col >= matrix.num_cols) {
                out_vec[row] = 0;
                return;
            }
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
    int row = 1 << 6;
    int col = 1 << 6;
    int nnz = 1 << 2;

    CSRMatrix<int> csr_matrix;
    csr_matrix.num_rows = row;
    csr_matrix.num_cols = col;
    csr_matrix.num_none_zeros = nnz;
    auto row_ptr_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix.num_rows + 1));
    auto col_indices_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix.num_none_zeros));
    auto values_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix.num_none_zeros));
    csr_matrix.row_ptr = row_ptr_buf.data();
    csr_matrix.col_indices = col_indices_buf.data();
    csr_matrix.values = values_buf.data();
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
        csr_matrix.col_indices[i] = i % col;
    }
    assert(validate_csr(csr_matrix));

    auto in_vec_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(col));
    int* in_vec = in_vec_buf.data();
    for (int i = 0; i < col; i++) {
        in_vec[i] = i % 1000;
    }

    auto out_vec_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(row));
    int* out_vec = out_vec_buf.data();

    CSRMatrix<int> csr_matrix_d;
    csr_matrix_d.num_rows = csr_matrix.num_rows;
    csr_matrix_d.num_cols = csr_matrix.num_cols;
    csr_matrix_d.num_none_zeros = csr_matrix.num_none_zeros;
    auto row_ptr_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix_d.num_rows + 1));
    auto col_indices_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix_d.num_none_zeros));
    auto values_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(csr_matrix_d.num_none_zeros));
    csr_matrix_d.row_ptr = row_ptr_d_buf.data();
    csr_matrix_d.col_indices = col_indices_d_buf.data();
    csr_matrix_d.values = values_d_buf.data();
    values_d_buf.copy_from_host(csr_matrix.values, static_cast<std::size_t>(csr_matrix.num_none_zeros));
    row_ptr_d_buf.copy_from_host(csr_matrix.row_ptr, static_cast<std::size_t>(csr_matrix.num_rows + 1));
    col_indices_d_buf.copy_from_host(csr_matrix.col_indices, static_cast<std::size_t>(csr_matrix.num_none_zeros));

    auto in_vec_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(col));
    int* in_vec_d = in_vec_d_buf.data();
    in_vec_d_buf.copy_from_host(in_vec, static_cast<std::size_t>(col));

    auto out_vec_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(row));
    int* out_vec_d = out_vec_d_buf.data();

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(row));
    int* expected = expected_buf.data();
    memset(expected, 0, row * sizeof(int));
    cpu_spmv_csr(expected, csr_matrix, in_vec);

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- SpMV using CSR --------------
    timer.start();
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (csr_matrix.num_rows + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
		out_vec_d_buf.fill_zero();
        spmv_csr<<<numBlocks, numThreadsPerBlock>>>(out_vec_d, csr_matrix_d, in_vec_d);
        CUDA_OK(cudaGetLastError());
    }
    ms = timer.elapsed_ms();
    printf("[CSR] Average time per SpMV: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    out_vec_d_buf.copy_to_host(out_vec, static_cast<std::size_t>(row));
    out_vec_d_buf.synchronize();

    // Verify result
    assert(verify_result(out_vec, expected, row));
    printf("SpMV CSR completed successfully.\n");
    return 0;
}
