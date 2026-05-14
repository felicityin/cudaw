#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "exception.cuh"

// Fixed-size device-side buffer
template <typename T>
class CudaBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                "CudaBuffer<T> requires trivially copyable T");

 public:
    CudaBuffer() = default;

    static CudaBuffer with_capacity(std::size_t capacity,
                                    cudaStream_t stream = nullptr) {
        CudaBuffer buf;
        buf.stream_ = stream;
        buf.cap_ = capacity;
        buf.len_ = 0;
        if (capacity == 0) {
          buf.ptr_ = nullptr;
          buf.owns_ = true;
          return buf;
        }
        buf.ptr_ = allocate_(capacity, stream);
        buf.owns_ = true;
		buf.len_ = capacity;
        return buf;
    }

    static CudaBuffer null(cudaStream_t stream = nullptr) {
        CudaBuffer buf;
        buf.ptr_ = nullptr;
        buf.len_ = 0;
        buf.cap_ = 0;
        buf.stream_ = stream;
        buf.owns_ = false;
        return buf;
    }

    // # Safety:
    // - ptr must point to device memory of at least capacity * sizeof(T) bytes.
    // - len <= capacity.
    // - If owns=true, ptr must be allocated by allocate_() compatible allocator.
    static CudaBuffer from_raw_parts(T* ptr,
                                    std::size_t len,
                                    std::size_t capacity,
                                    cudaStream_t stream = nullptr,
                                    bool owns = false) {
        assert(len <= capacity);
        CudaBuffer buf;
        buf.ptr_ = ptr;
        buf.len_ = len;
        buf.cap_ = capacity;
        buf.stream_ = stream;
        buf.owns_ = owns;
        return buf;
    }

	~CudaBuffer() { release(); }

	CudaBuffer(const CudaBuffer&) = delete;
	CudaBuffer& operator=(const CudaBuffer&) = delete;

	CudaBuffer(CudaBuffer&& other) noexcept { move_from_(other); }

	CudaBuffer& operator=(CudaBuffer&& other) noexcept {
		if (this == &other) return *this;
		release();
		move_from_(other);
		return *this;
  	}

  	// Releases device memory (if owned) and resets to null/empty.
	void release() noexcept {
		if (owns_ && ptr_ != nullptr) {
			free_(ptr_, stream_);
		}
		ptr_ = nullptr;
		len_ = 0;
		cap_ = 0;
		stream_ = nullptr;
		owns_ = false;
	}

	// Sets all bytes in the allocation to 0 via cudaMemset (synchronous with
	// respect to the legacy default stream semantics).
	void fill_zero() {
		if (ptr_ == nullptr || cap_ == 0) return;
		CUDA_OK(cudaMemset(ptr_, 0, cap_ * sizeof(T)));
	}

	const T* data() const noexcept { return ptr_; }
	T* data() noexcept { return ptr_; }

    std::size_t len() const noexcept { return len_; }
    std::size_t capacity() const noexcept { return cap_; }
    bool empty() const noexcept { return len_ == 0; }
	bool is_null() const noexcept { return ptr_ == nullptr; }
	cudaStream_t stream() const noexcept { return stream_; }
	void synchronize() const { CUDA_OK(cudaStreamSynchronize(stream_)); }

    // # Safety: caller guarantees stream lifetime & ordering correctness.
    void set_stream(cudaStream_t stream) noexcept { stream_ = stream; }

    // # Safety: caller guarantees new_len <= capacity().
    void set_len(std::size_t new_len) noexcept {
        assert(new_len <= cap_);
        len_ = new_len;
    }

    // # Safety: caller guarantees the whole allocation is initialized when used.
    void set_max_len() noexcept { len_ = cap_; }

    // Set buffer bytes from [0, len * sizeof(T)) to a fixed value.
    void set_bytes(unsigned char value) {
        if (ptr_ == nullptr || len_ == 0) return;
        CUDA_OK(cudaMemsetAsync(ptr_, static_cast<int>(value), len_ * sizeof(T),
                                stream_));
    }

	// Copy exactly len() elements from host to device.
	void copy_from_host(const T* host_src, std::size_t count) {
		assert(count == len_);
		if (count == 0) return;
		assert(ptr_ != nullptr);
		CUDA_OK(cudaMemcpyAsync(ptr_, host_src, count * sizeof(T),
								cudaMemcpyHostToDevice, stream_));
	}

	// Copy exactly len() elements from device to host.
	void copy_to_host(T* host_dst, std::size_t count) const {
		assert(count == len_);
		if (count == 0) return;
		assert(ptr_ != nullptr);
		CUDA_OK(cudaMemcpyAsync(host_dst, ptr_, count * sizeof(T),
								cudaMemcpyDeviceToHost, stream_));
	}

	// Append elements from host into the remaining capacity.
	void extend_from_host(const T* host_src, std::size_t count) {
		assert(len_ + count <= cap_);
		if (count == 0) return;
		assert(ptr_ != nullptr);
		CUDA_OK(cudaMemcpyAsync(ptr_ + len_, host_src, count * sizeof(T),
								cudaMemcpyHostToDevice, stream_));
		len_ += count;
	}

    // Append elements from another device buffer/slice.
    void extend_from_device(const T* device_src, std::size_t count) {
        assert(len_ + count <= cap_);
        if (count == 0) return;
        assert(ptr_ != nullptr);
        CUDA_OK(cudaMemcpyAsync(ptr_ + len_, device_src, count * sizeof(T),
                                cudaMemcpyDeviceToDevice, stream_));
        len_ += count;
    }

private:
    static T* allocate_(std::size_t count, cudaStream_t stream) {
        void* p = nullptr;
    #if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        // cudaMallocAsync is available since CUDA 11.2.
        CUDA_OK(cudaMallocAsync(&p, count * sizeof(T), stream));
    #else
        (void)stream;
        CUDA_OK(cudaMalloc(&p, count * sizeof(T)));
    #endif
        return static_cast<T*>(p);
    }

    static void free_(T* ptr, cudaStream_t stream) noexcept {
        if (ptr == nullptr) return;
    #if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        (void)cudaFreeAsync(ptr, stream);
    #else
        (void)stream;
        (void)cudaFree(ptr);
    #endif
    }

    void move_from_(CudaBuffer& other) noexcept {
        ptr_ = other.ptr_;
        len_ = other.len_;
        cap_ = other.cap_;
        stream_ = other.stream_;
        owns_ = other.owns_;

        other.ptr_ = nullptr;
        other.len_ = 0;
        other.cap_ = 0;
        other.stream_ = nullptr;
        other.owns_ = false;
    }

    T* ptr_{nullptr};
    std::size_t len_{0};
    std::size_t cap_{0};
    cudaStream_t stream_{nullptr};
    bool owns_{false};
};
