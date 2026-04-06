#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

__global__ void rgb2gray(unsigned char* red, unsigned char* green, unsigned char* blue,
                         unsigned char* gray, unsigned int width, unsigned int height) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Convert RGB to grayscale using the luminosity method
    if (row < height && col < width) {
        unsigned int idx = row * width + col;
        gray[idx] = red[idx] * 0.299 + green[idx] * 0.587 + blue[idx] * 0.114;
    }
}

#define BLUR_RADIUS 1

__global__ void blur(unsigned char* image, unsigned char* blurred,
                     unsigned int width, unsigned int height) {
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outRow < height && outCol < width) { 
        unsigned int average = 0;
        for (int row = outRow - BLUR_RADIUS; row <= outRow + BLUR_RADIUS; row++) {
            for (int col = outCol - BLUR_RADIUS; col <= outCol + BLUR_RADIUS; col++) {
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    average += image[row * width + col];
                }
            }
        }
        blurred[outRow * width + outCol] = average / ((2 * BLUR_RADIUS + 1) * (2 * BLUR_RADIUS + 1));
    }
}

#define TILE_DIM 32

__global__ void mm_tiled_kernel(float* C, float* A, float* B, int n) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;

    for (int tile = 0; tile < n / TILE_DIM; tile++) {
        A_s[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * n + col] = sum;
}

#define COARSE_FACTOR 4

__global__ void mm_coarse_tiled_kernel(float* C, float* A, float* B, int n) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int colStart = blockIdx.x * blockDim.x * COARSE_FACTOR + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum[COARSE_FACTOR] = {0.0f};

    for (int tile = 0; tile < n / TILE_DIM; tile++) {
        A_s[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_DIM + threadIdx.x];
        
        for (unsigned int i = 0; i < COARSE_FACTOR; i++) {
            unsigned int col = colStart + i * TILE_DIM;
            B_s[threadIdx.y][threadIdx.x] = B[(tile * TILE_DIM + threadIdx.y) * n + col];
            __syncthreads();

            for (int k = 0; k < TILE_DIM; k++) {
                sum += A_s[threadIdx.y][k] * B_s[k][threadIdx.x];
            }
            __syncthreads();
        }
    }

    C[row * n + col] = sum;
}
