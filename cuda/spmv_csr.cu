#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cstring>

int NUM_REPS = 100;

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

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
    csr_matrix.row_ptr = (int*)malloc((csr_matrix.num_rows + 1) * sizeof(int));
    csr_matrix.col_indices = (int*)malloc(csr_matrix.num_none_zeros * sizeof(int));
    csr_matrix.values = (int*)malloc(csr_matrix.num_none_zeros * sizeof(int));
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

    int* in_vec = (int*)malloc(col * sizeof(int));
    for (int i = 0; i < col; i++) {
        in_vec[i] = i % 1000;
    }

    int* out_vec = (int*)malloc(row * sizeof(int));

    CSRMatrix<int> csr_matrix_d;
    csr_matrix_d.num_rows = csr_matrix.num_rows;
    csr_matrix_d.num_cols = csr_matrix.num_cols;
    csr_matrix_d.num_none_zeros = csr_matrix.num_none_zeros;
    CUDA_OK(cudaMalloc((void**)&csr_matrix_d.row_ptr, (csr_matrix_d.num_rows + 1) * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&csr_matrix_d.col_indices, csr_matrix_d.num_none_zeros * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&csr_matrix_d.values, csr_matrix_d.num_none_zeros * sizeof(int)));
    CUDA_OK(cudaMemcpy(csr_matrix_d.values, csr_matrix.values, csr_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(csr_matrix_d.row_ptr, csr_matrix.row_ptr, (csr_matrix.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(csr_matrix_d.col_indices, csr_matrix.col_indices, csr_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));

    int* in_vec_d;
    CUDA_OK(cudaMalloc((void**)&in_vec_d, col * sizeof(int)));
    CUDA_OK(cudaMemcpy(in_vec_d, in_vec, col * sizeof(int), cudaMemcpyHostToDevice));

    int* out_vec_d;
    CUDA_OK(cudaMalloc((void**)&out_vec_d, row * sizeof(int)));

    int* expected = (int*)malloc(row * sizeof(int));
    memset(expected, 0, row * sizeof(int));
    cpu_spmv_csr(expected, csr_matrix, in_vec);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- SpMV using CSR --------------
    CUDA_OK(cudaEventRecord(startEvent));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (csr_matrix.num_rows + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(out_vec_d, 0, row * sizeof(int)));
        spmv_csr<<<numBlocks, numThreadsPerBlock>>>(out_vec_d, csr_matrix_d, in_vec_d);
        CUDA_OK(cudaGetLastError());
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[CSR] Average time per SpMV: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(out_vec, out_vec_d, row * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(out_vec, expected, row));
    printf("SpMV CSR completed successfully.\n");

    free(csr_matrix.values);
    free(csr_matrix.row_ptr);
    free(csr_matrix.col_indices);
    free(in_vec);
    free(out_vec);
    free(expected);
    CUDA_OK(cudaFree(csr_matrix_d.values));
    CUDA_OK(cudaFree(csr_matrix_d.row_ptr));
    CUDA_OK(cudaFree(csr_matrix_d.col_indices));
    CUDA_OK(cudaFree(in_vec_d));
    CUDA_OK(cudaFree(out_vec_d));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));
    return 0;
}
