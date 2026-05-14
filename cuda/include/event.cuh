#pragma once

#include <cuda_runtime.h>

#include "exception.cuh"

class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDefault) {
        CUDA_OK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (event_ != nullptr) {
            (void)cudaEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
        return *this;
    }

    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream = nullptr) { CUDA_OK(cudaEventRecord(event_, stream)); }

    void synchronize() { CUDA_OK(cudaEventSynchronize(event_)); }

    static float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
        float ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms;
    }

private:
    cudaEvent_t event_{nullptr};
};
