#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "event.cuh"
#include "exception.cuh"

// CUDA stream RAII wrapper.
//
// - Default-constructs to the CUDA default stream (nullptr) and does not own it.
// - create() allocates a new stream and owns/destroys it.
// - Provides common stream-ordered operations (sync, wait on event, async alloc/copies, host callbacks).
class CudaStream {
public:
	// Default stream (nullptr). Not owned.
	CudaStream() = default;

	static CudaStream create(unsigned int flags = cudaStreamDefault) {
		cudaStream_t s = nullptr;
		CUDA_OK(cudaStreamCreateWithFlags(&s, flags));
		return CudaStream(s, /*owns=*/true);
	}

	static CudaStream null() { return CudaStream(nullptr, /*owns=*/false); }

	~CudaStream() { release(); }

	CudaStream(const CudaStream&) = delete;
	CudaStream& operator=(const CudaStream&) = delete;

	CudaStream(CudaStream&& other) noexcept { move_from_(other); }

	CudaStream& operator=(CudaStream&& other) noexcept {
		if (this == &other) return *this;
		release();
		move_from_(other);
		return *this;
	}

	void release() noexcept {
		if (owns_ && stream_ != nullptr) {
			(void)cudaStreamDestroy(stream_);
		}
		stream_ = nullptr;
		owns_ = false;
	}

	cudaStream_t get() const noexcept { return stream_; }
	bool is_default() const noexcept { return stream_ == nullptr; }
	bool owns() const noexcept { return owns_; }

	void synchronize() const { CUDA_OK(cudaStreamSynchronize(stream_)); }

	void record(CudaEvent& event) const { event.record(stream_); }

	void wait_event(const CudaEvent& event, unsigned int flags = 0) const {
		CUDA_OK(cudaStreamWaitEvent(stream_, event.get(), flags));
	}

	// Enqueue a host callback to run after all previously enqueued work in this stream completes.
	//
	// The callback must be a plain function pointer + opaque context pointer.
	void launch_host_fn(cudaHostFn_t host_fn, void* user_data) const {
		CUDA_OK(cudaLaunchHostFunc(stream_, host_fn, user_data));
	}

	// Async device allocation/free ordered on this stream.
	template <typename T>
	T* malloc_async(std::size_t count) const {
		static_assert(std::is_trivially_copyable_v<T>,
					  "CudaStream::malloc_async requires trivially copyable T");
		void* p = nullptr;
	#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
		CUDA_OK(cudaMallocAsync(&p, count * sizeof(T), stream_));
	#else
		(void)stream_;
		CUDA_OK(cudaMalloc(&p, count * sizeof(T)));
	#endif
		return static_cast<T*>(p);
	}

	template <typename T>
	void free_async(T* ptr) const noexcept {
		if (ptr == nullptr) return;
	#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
		(void)cudaFreeAsync(ptr, stream_);
	#else
		(void)stream_;
		(void)cudaFree(ptr);
	#endif
	}

	template <typename T>
	void memset_async(T* dst, unsigned char value, std::size_t count) const {
		static_assert(std::is_trivially_copyable_v<T>,
					  "CudaStream::memset_async requires trivially copyable T");
		if (dst == nullptr || count == 0) return;
		CUDA_OK(
			cudaMemsetAsync(dst, static_cast<int>(value), count * sizeof(T), stream_));
	}

	template <typename T>
	void memcpy_device_to_device_async(T* dst, const T* src, std::size_t count) const {
		static_assert(std::is_trivially_copyable_v<T>,
					  "CudaStream::memcpy_device_to_device_async requires trivially copyable T");
		if (count == 0) return;
		assert(dst != nullptr && src != nullptr);
		CUDA_OK(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice,
								stream_));
	}

	template <typename T>
	void memcpy_host_to_device_async(T* dst, const T* src, std::size_t count) const {
		static_assert(std::is_trivially_copyable_v<T>,
					  "CudaStream::memcpy_host_to_device_async requires trivially copyable T");
		if (count == 0) return;
		assert(dst != nullptr && src != nullptr);
		CUDA_OK(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice,
								stream_));
	}

	template <typename T>
	void memcpy_device_to_host_async(T* dst, const T* src, std::size_t count) const {
		static_assert(std::is_trivially_copyable_v<T>,
					  "CudaStream::memcpy_device_to_host_async requires trivially copyable T");
		if (count == 0) return;
		assert(dst != nullptr && src != nullptr);
		CUDA_OK(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost,
								stream_));
	}

	template <typename T>
	void memcpy_host_to_host_async(T* dst, const T* src, std::size_t count) const {
		static_assert(std::is_trivially_copyable_v<T>,
					  "CudaStream::memcpy_host_to_host_async requires trivially copyable T");
		if (count == 0) return;
		assert(dst != nullptr && src != nullptr);
		CUDA_OK(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToHost,
								stream_));
	}

private:
	explicit CudaStream(cudaStream_t stream, bool owns) : stream_(stream), owns_(owns) {}

	void move_from_(CudaStream& other) noexcept {
		stream_ = other.stream_;
		owns_ = other.owns_;
		other.stream_ = nullptr;
		other.owns_ = false;
	}

	cudaStream_t stream_{nullptr};
	bool owns_{false};
};
