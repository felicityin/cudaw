#pragma once

#include <cuda_runtime.h>

#include "event.cuh"

class CudaTimer {
public:
    CudaTimer() = default;

    void start(cudaStream_t stream = nullptr) { start_.record(stream); }

    float elapsed_ms(cudaStream_t stream = nullptr) {
        stop_.record(stream);
        stop_.synchronize();
        return CudaEvent::elapsed_ms(start_, stop_);
    }

private:
    CudaEvent start_{};
    CudaEvent stop_{};
};
