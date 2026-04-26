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
        csr_matrix.col_indices[i] = i % cols;
    }

    ELLMatrix<int> ell_matrix;
    ell_matrix.num_rows = rows;
    ell_matrix.num_cols = cols;
    ell_matrix.nnz_per_row = (int*)malloc(rows * sizeof(int));

    int max_nnz_per_row = 0;
    for (int i = 0; i < rows; ++i) {
        int nnz = csr_matrix.row_ptr[i + 1] - csr_matrix.row_ptr[i];
        ell_matrix.nnz_per_row[i] = nnz;
        max_nnz_per_row = max_nnz_per_row < nnz ? nnz : max_nnz_per_row;
    }
    ell_matrix.max_nnz_per_row = max_nnz_per_row;
    ell_matrix.col_indices = (int*)malloc(max_nnz_per_row * rows * sizeof(int));
    ell_matrix.values = (int*)malloc(max_nnz_per_row * rows * sizeof(int));

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

    int* in_vec = (int*)malloc(cols * sizeof(int));
    for (int i = 0; i < cols; i++) {
        in_vec[i] = i % 1000;
    }

    int* out_vec = (int*)malloc(rows * sizeof(int));

    ELLMatrix<int> ell_matrix_d;
    ell_matrix_d.num_rows = ell_matrix.num_rows;
    ell_matrix_d.num_cols = ell_matrix.num_cols;
    ell_matrix_d.max_nnz_per_row = ell_matrix.max_nnz_per_row;
    CUDA_OK(cudaMalloc((void**)&ell_matrix_d.nnz_per_row, rows * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&ell_matrix_d.col_indices, max_nnz_per_row * rows * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&ell_matrix_d.values, max_nnz_per_row * rows * sizeof(int)));
    CUDA_OK(cudaMemcpy(ell_matrix_d.nnz_per_row, ell_matrix.nnz_per_row, rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ell_matrix_d.values, ell_matrix.values, max_nnz_per_row * rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ell_matrix_d.col_indices, ell_matrix.col_indices, max_nnz_per_row * rows * sizeof(int), cudaMemcpyHostToDevice));

    int* in_vec_d;
    CUDA_OK(cudaMalloc((void**)&in_vec_d, cols * sizeof(int)));
    CUDA_OK(cudaMemcpy(in_vec_d, in_vec, cols * sizeof(int), cudaMemcpyHostToDevice));

    int* out_vec_d;
    CUDA_OK(cudaMalloc((void**)&out_vec_d, rows * sizeof(int)));

    int* expected = (int*)malloc(rows * sizeof(int));
    memset(expected, 0, rows * sizeof(int));
    cpu_spmv_ell(expected, ell_matrix, in_vec);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- SpMV using ELL --------------
    CUDA_OK(cudaEventRecord(startEvent));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (ell_matrix.num_rows + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(out_vec_d, 0, rows * sizeof(int)));
        spmv_ell<<<numBlocks, numThreadsPerBlock>>>(out_vec_d, ell_matrix_d, in_vec_d);
        CUDA_OK(cudaGetLastError());
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[ELL] Average time per SpMV: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(out_vec, out_vec_d, rows * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(out_vec, expected, rows));
    printf("SpMV ELL completed successfully.\n");

    free(csr_matrix.values);
    free(csr_matrix.row_ptr);
    free(csr_matrix.col_indices);
    free(ell_matrix.values);
    free(ell_matrix.nnz_per_row);
    free(ell_matrix.col_indices);
    free(in_vec);
    free(out_vec);
    free(expected);
    CUDA_OK(cudaFree(ell_matrix_d.values));
    CUDA_OK(cudaFree(ell_matrix_d.nnz_per_row));
    CUDA_OK(cudaFree(ell_matrix_d.col_indices));
    CUDA_OK(cudaFree(in_vec_d));
    CUDA_OK(cudaFree(out_vec_d));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));
    return 0;
}
