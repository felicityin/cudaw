#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

const int NUM_REPS = 100;

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

void cpu_matrix_multiply(int *output, const int *a, const int *b, int m, int n, int k) {
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < k; x++) {
            int tmp = 0;
            for (int step = 0; step < n; step++) {
                tmp += a[y * n + step] * b[step * k + x];
            }
            output[y * k + x] = tmp;
        }
    }
}

__global__ void multiplyNaive(int *output, const int *a, const int *b, int m, int n, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m && y < k) {
        int tmp = 0;
        for (int step = 0; step < n; step++) {
            tmp += a[y * n + step] * b[step * k + x];
        }
        output[y * k + x] = tmp;
    }
}

// ref https://blog.csdn.net/kunhe0512/article/details/131381155
// a[][] * b[][] = c[][]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31
// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
template <int BLOCK_SIZE>
__global__ void multiplySharedMemory(int *output, const int *a, const int *b, int m, int n, int k) {
    __shared__ int sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int sub_b[BLOCK_SIZE][BLOCK_SIZE];

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tmp = 0;

    for (int step = 0; step <= n; step += blockDim.x) {
        int step_y = y;
        int step_x = step + threadIdx.x;
        if (step_y < m && step_x < n) {
            sub_a[threadIdx.y][threadIdx.x] = a[step_y * n + step_x];
        } else {
            sub_a[threadIdx.y][threadIdx.x] = 0;
        }

        step_y = step + threadIdx.y;
        step_x = x;
        if (step_y < n && step_x < k) {
            sub_b[threadIdx.y][threadIdx.x] = b[step_y * k + step_x];
        } else {
            sub_b[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; i++) {
            tmp += sub_a[threadIdx.y][i] * sub_b[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (y < m && x < k) {
        output[y * k + x] = tmp;
    }
}

bool verify_result(const int *a, const int *b, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (a[y * width + x] != b[y * width + x]) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const int *matrix, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%d ", matrix[y * width + x]);
        }
        printf("\n");
    }
}

int main() {
    int m = 1 << 10;
    int n = 1 << 11;
    int k = 1 << 10;

    int *h_a = (int*)malloc(m * n * sizeof(int));
    int *h_b = (int*)malloc(n * k * sizeof(int));
    int *h_c = (int*)malloc(m * k * sizeof(int));

    int *d_a, *d_b, *d_c;
    CUDA_OK(cudaMalloc((void**)&d_a, m * n * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_b, n * k * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_c, m * k * sizeof(int)));

    // Initialize input matrix
    for (int i = 0; i < m * n; ++i) {
        h_a[i] = i;
    }
    for (int i = 0; i < n * k; ++i) {
        h_b[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_a, h_a, m * n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_b, h_b, n * k * sizeof(int), cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    float ms;

    // -------------- Multiplication using naive kernel --------------

    const int block_size = 32;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((k + block_size - 1) / block_size, (m + block_size - 1) / block_size);

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        multiplyNaive<<<dimGrid, dimBlock>>>(d_c, d_a, d_b, m, n, k);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[native] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_c, d_c, m * k * sizeof(int), cudaMemcpyDeviceToHost));

    int *c = (int*)malloc(m * k * sizeof(int));
    cpu_matrix_multiply(c, h_a, h_b, m, n, k);
    // print_matrix(c, m, k);
    // print_matrix(h_c, m, k);
    assert(verify_result(h_c, c, m, k));

    // -------------- Multiplication using shared memory kernel --------------

    const int BLOCK_SIZE = 32;
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    CUDA_OK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_REPS; i++) {
        multiplySharedMemory<BLOCK_SIZE><<<gridSize, blockSize>>>(d_c, d_a, d_b, m, n, k);
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[shared memory] Average time per multiplication: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_c, d_c, m * k * sizeof(int), cudaMemcpyDeviceToHost));

    // print_matrix(h_c, m, k);
    assert(verify_result(h_c, c, m, k));

    printf("Multiplication completed successfully.\n");

    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaFree(d_c));

    return 0;
}
