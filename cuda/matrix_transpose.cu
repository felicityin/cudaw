#include <stdio.h>
#include <assert.h>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int NUM_REPS = 100;

__global__ void transposeNaive(int *output, const int *input, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t len = width * height;

    size_t idx_in = row * width + col;
    size_t idx_out = col * height + row;
    if (idx_in < len && idx_out < len) {
        output[idx_out] = input[idx_in];
    }
}

__global__ void transposeNaiveV2(int *output, const int *input, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t len = width * height;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = (row + j) * width + col;
        size_t idx_out = col * height + row + j;
        if (idx_in < len && idx_out < len) {
            output[idx_out] = input[idx_in];
        }
    }
}

__global__ void transposeSharedMemory(int *output, const int *input, int width, int height) {
    __shared__ int s_mem[TILE_DIM][TILE_DIM+1];
        
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t len = width * height;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_in = (row + j) * width + col;
        if (idx_in < len) {
            s_mem[threadIdx.y + j][threadIdx.x] = input[idx_in];
        }
    }

    __syncthreads();

    col = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    row = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        size_t idx_out = (row + j) * height + col;
        if (idx_out < len) {
            output[idx_out] = s_mem[threadIdx.x][threadIdx.y + j];
        }
    }
}

__global__ void __launch_bounds__(TILE_DIM)
transposeSharedMemoryV2(int *output, const int *input, int width, int height) {
    __shared__ int s_mem[TILE_DIM][TILE_DIM + 1];

    size_t dim_x = (width + TILE_DIM - 1) / TILE_DIM;
    size_t bid = blockIdx.x; // (x, 1, 1)
    size_t bid_y = bid / dim_x;
    size_t bid_x = bid % dim_x; // (bid_x, bid_y, 1)

    size_t tid = threadIdx.x;
    size_t idx_in = bid_y * TILE_DIM * width + bid_x * TILE_DIM + tid;
    size_t idx_out = bid_x * TILE_DIM * height + bid_y * TILE_DIM + tid;

    bool boundray_column = bid_x * TILE_DIM + tid < width;
    size_t row_offset = bid_y * TILE_DIM + 0;
    for (auto i = 0; i < TILE_DIM; ++i) {
        bool boundray = boundray_column && (row_offset + i < height);
        s_mem[i][tid] = (boundray) ? input[idx_in + i * width] : 0;
    }

    __syncthreads();

    boundray_column = bid_y * TILE_DIM + tid < height;
    row_offset = bid_x * TILE_DIM + 0;
    for (auto i = 0; i < TILE_DIM; ++i) {
        bool boundray = boundray_column && (row_offset + i < width);
        if (boundray)
            output[idx_out + i * height] = s_mem[tid][i];
    }
}

bool verify_result(const int *h_output, const int *h_input, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (h_output[j * height + i] != h_input[i * width + j]) {
                return false;
            }
        }
    }
    return true;
}

void print_matrix(const int *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%d ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int width = 100;
    int height = 1 << 20;
    const size_t size = width * height * sizeof(int);

    auto h_input_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(width) * height);
    auto h_output_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(width) * height);
    int* h_input = h_input_buf.data();
    int* h_output = h_output_buf.data();

    auto d_input_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(width) * height);
    auto d_output_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(width) * height);
    int* d_input = d_input_buf.data();
    int* d_output = d_output_buf.data();

    // Initialize input matrix
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = i;
    }

    CUDA_OK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    CudaTimer timer;
    float ms = 0.0f;

    // -------------- Transpose using naive kernel --------------

    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        transposeNaive<<<dimGrid, dimBlock>>>(d_output, d_input, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[native] Average time per transpose: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    assert(verify_result(h_output, h_input, width, height));

    // -------------- Transpose using naive kernel v2 --------------

    dim3 blockSize(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        transposeNaiveV2<<<gridSize, blockSize>>>(d_output, d_input, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[native v2] Average time per transpose: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    assert(verify_result(h_output, h_input, width, height));

    // -------------- Transpose using shared memory kernel --------------

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        transposeSharedMemory<<<gridSize, blockSize>>>(d_output, d_input, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory] Average time per transpose: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    assert(verify_result(h_output, h_input, width, height));

    // -------------- Transpose using shared memory kernel v2 --------------

    size_t grid_x = (width + TILE_DIM - 1) / TILE_DIM;
    size_t grid_y = (height + TILE_DIM - 1) / TILE_DIM;

    dim3 grid(grid_x * grid_y);
    dim3 block(TILE_DIM);

    timer.start();

    for (int i = 0; i < NUM_REPS; i++) {
        transposeSharedMemoryV2<<<grid, block>>>(d_output, d_input, width, height);
    }
    ms = timer.elapsed_ms();
    printf("[shared memory v2] Average time per transpose: %f ms\n", ms / NUM_REPS);

    CUDA_OK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    assert(verify_result(h_output, h_input, width, height));

    printf("Transpose completed successfully.\n");

    return 0;
}
