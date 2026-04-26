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
    coo_matrix.num_none_zeros = 1 << 2;
    coo_matrix.values = (int*)malloc(coo_matrix.num_none_zeros * sizeof(int));
    coo_matrix.row_indices = (int*)malloc(coo_matrix.num_none_zeros * sizeof(int));
    coo_matrix.col_indices = (int*)malloc(coo_matrix.num_none_zeros * sizeof(int));
    for (int i = 0; i < coo_matrix.num_none_zeros; i++) {
        coo_matrix.values[i] = i % 1000;
        coo_matrix.row_indices[i] = i % row;
        coo_matrix.col_indices[i] = i % col;

    }

    int* in_vec = (int*)malloc(col * sizeof(int));
    for (int i = 0; i < col; i++) {
        in_vec[i] = i % 1000;
    }

    int* out_vec = (int*)malloc(col * sizeof(int));

    COOMatrix<int> coo_matrix_d;
    coo_matrix_d.num_rows = coo_matrix.num_rows;
    coo_matrix_d.num_cols = coo_matrix.num_cols;
    coo_matrix_d.num_none_zeros = coo_matrix.num_none_zeros;
    CUDA_OK(cudaMalloc((void**)&coo_matrix_d.values, coo_matrix.num_none_zeros * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&coo_matrix_d.row_indices, coo_matrix.num_none_zeros * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&coo_matrix_d.col_indices, coo_matrix.num_none_zeros * sizeof(int)));
    CUDA_OK(cudaMemcpy(coo_matrix_d.values, coo_matrix.values, coo_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(coo_matrix_d.row_indices, coo_matrix.row_indices, coo_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(coo_matrix_d.col_indices, coo_matrix.col_indices, coo_matrix.num_none_zeros * sizeof(int), cudaMemcpyHostToDevice));

    int* in_vec_d;
    CUDA_OK(cudaMalloc((void**)&in_vec_d, col * sizeof(int)));
    CUDA_OK(cudaMemcpy(in_vec_d, in_vec, col * sizeof(int), cudaMemcpyHostToDevice));

    int* out_vec_d;
    CUDA_OK(cudaMalloc((void**)&out_vec_d, col * sizeof(int)));

    int* expected = (int*)malloc(col * sizeof(int));
    memset(expected, 0, col * sizeof(int));
    cpu_spmv_coo(expected, coo_matrix, in_vec);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    // -------------- SpMV using COO --------------
    CUDA_OK(cudaEventRecord(startEvent));
    unsigned int numThreadsPerBlock = 1024;
    unsigned int numBlocks = (coo_matrix_d.num_none_zeros + numThreadsPerBlock - 1) / numThreadsPerBlock;

    for (int i = 0; i < NUM_REPS; i++) {
        CUDA_OK(cudaMemset(out_vec_d, 0, row * sizeof(int)));
        spmv_coo<<<numBlocks, numThreadsPerBlock>>>(out_vec_d, coo_matrix_d, in_vec_d);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[COO] Average time per SpMV: %f ms\n", ms / NUM_REPS);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(out_vec, out_vec_d, col * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(out_vec, expected, row));
    printf("SpMV COO completed successfully.\n");

    free(coo_matrix.values);
    free(coo_matrix.row_indices);
    free(coo_matrix.col_indices);
    free(in_vec);
    free(out_vec);
    free(expected);
    CUDA_OK(cudaFree(coo_matrix_d.values));
    CUDA_OK(cudaFree(coo_matrix_d.row_indices));
    CUDA_OK(cudaFree(coo_matrix_d.col_indices));
    CUDA_OK(cudaFree(in_vec_d));
    CUDA_OK(cudaFree(out_vec_d));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));;
    return 0;
}
