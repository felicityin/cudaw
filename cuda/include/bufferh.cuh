#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "exception.cuh"

// Fixed-size pinned host-side buffer (RAII).
//
// Notes:
// - Intended for trivially copyable types, similar to CudaBuffer<T>.
// - Uses cudaMallocHost/cudaFreeHost (pinned/page-locked host memory).
template <typename T>
class PinBuffer {
	static_assert(std::is_trivially_copyable_v<T>,
				"PinBuffer<T> requires trivially copyable T");

public:
	PinBuffer() = default;

	static PinBuffer with_capacity(std::size_t capacity) {
		PinBuffer buf;
		buf.cap_ = capacity;
		buf.len_ = 0;
		if (capacity == 0) {
			buf.ptr_ = nullptr;
			buf.owns_ = true;
			return buf;
		}
		buf.ptr_ = allocate_(capacity);
		buf.owns_ = true;
		buf.len_ = capacity;
		return buf;
	}

	static PinBuffer null() {
		PinBuffer buf;
		buf.ptr_ = nullptr;
		buf.len_ = 0;
		buf.cap_ = 0;
		buf.owns_ = false;
		return buf;
	}

	// # Safety:
	// - ptr must point to pinned host memory of at least capacity * sizeof(T)
	//   bytes.
	// - len <= capacity.
	// - If owns=true, ptr must be allocated by allocate_() compatible allocator.
	static PinBuffer from_raw_parts(T* ptr,
									std::size_t len,
									std::size_t capacity,
									bool owns = false) {
		assert(len <= capacity);
		PinBuffer buf;
		buf.ptr_ = ptr;
		buf.len_ = len;
		buf.cap_ = capacity;
		buf.owns_ = owns;
		return buf;
	}

	~PinBuffer() { release(); }

	PinBuffer(const PinBuffer&) = delete;
	PinBuffer& operator=(const PinBuffer&) = delete;

	PinBuffer(PinBuffer&& other) noexcept { move_from_(other); }

	PinBuffer& operator=(PinBuffer&& other) noexcept {
		if (this == &other) return *this;
		release();
		move_from_(other);
		return *this;
	}

	// Releases pinned host memory (if owned) and resets to null/empty.
	void release() noexcept {
		if (owns_ && ptr_ != nullptr) {
			free_(ptr_);
		}
		ptr_ = nullptr;
		len_ = 0;
		cap_ = 0;
		owns_ = false;
	}

	// Sets all bytes in the allocation to 0.
	void reset() {
		if (ptr_ == nullptr || cap_ == 0) return;
		std::memset(ptr_, 0, cap_ * sizeof(T));
	}

	const T* data() const noexcept { return ptr_; }
	T* data() noexcept { return ptr_; }

	std::size_t len() const noexcept { return len_; }
	std::size_t capacity() const noexcept { return cap_; }
	bool empty() const noexcept { return len_ == 0; }
	bool is_null() const noexcept { return ptr_ == nullptr; }

	// # Safety: caller guarantees new_len <= capacity().
	void set_len(std::size_t new_len) noexcept {
		assert(new_len <= cap_);
		len_ = new_len;
	}

	// # Safety: caller guarantees the whole allocation is initialized when used.
	void set_max_len() noexcept { len_ = cap_; }

	T& operator[](std::size_t idx) noexcept {
		assert(idx < len_);
		return ptr_[idx];
	}
	const T& operator[](std::size_t idx) const noexcept {
		assert(idx < len_);
		return ptr_[idx];
	}

private:
	static T* allocate_(std::size_t count) {
		void* p = nullptr;
		CUDA_OK(cudaMallocHost(&p, count * sizeof(T)));
		return static_cast<T*>(p);
	}

	static void free_(T* ptr) noexcept {
		if (ptr == nullptr) return;
		(void)cudaFreeHost(ptr);
}

	void move_from_(PinBuffer& other) noexcept {
		ptr_ = other.ptr_;
		len_ = other.len_;
		cap_ = other.cap_;
		owns_ = other.owns_;

		other.ptr_ = nullptr;
		other.len_ = 0;
		other.cap_ = 0;
		other.owns_ = false;
	}

	T* ptr_{nullptr};
	std::size_t len_{0};
	std::size_t cap_{0};
	bool owns_{false};
};


// Fixed-size host-side buffer (RAII).
//
// Notes:
// - Intended for trivially copyable types, similar to CudaBuffer<T>.
// - Uses malloc/free (not pinned memory).
template <typename T>
class HostBuffer {
static_assert(std::is_trivially_copyable_v<T>,
				"HostBuffer<T> requires trivially copyable T");

public:
	HostBuffer() = default;

	static HostBuffer with_capacity(std::size_t capacity) {
		HostBuffer buf;
		buf.cap_ = capacity;
		buf.len_ = 0;
		if (capacity == 0) {
			buf.ptr_ = nullptr;
			buf.owns_ = true;
			return buf;
		}
		buf.ptr_ = allocate_(capacity);
		buf.owns_ = true;
		buf.len_ = capacity;
		return buf;
	}

	static HostBuffer null() {
		HostBuffer buf;
		buf.ptr_ = nullptr;
		buf.len_ = 0;
		buf.cap_ = 0;
		buf.owns_ = false;
		return buf;
	}

	// # Safety:
	// - ptr must point to host memory of at least capacity * sizeof(T) bytes.
	// - len <= capacity.
	// - If owns=true, ptr must be allocated by allocate_() compatible allocator.
	static HostBuffer from_raw_parts(T* ptr,
									std::size_t len,
									std::size_t capacity,
									bool owns = false) {
		assert(len <= capacity);
		HostBuffer buf;
		buf.ptr_ = ptr;
		buf.len_ = len;
		buf.cap_ = capacity;
		buf.owns_ = owns;
		return buf;
	}

	~HostBuffer() { release(); }

	HostBuffer(const HostBuffer&) = delete;
	HostBuffer& operator=(const HostBuffer&) = delete;

	HostBuffer(HostBuffer&& other) noexcept { move_from_(other); }

	HostBuffer& operator=(HostBuffer&& other) noexcept {
		if (this == &other) return *this;
		release();
		move_from_(other);
		return *this;
	}

	// Releases host memory (if owned) and resets to null/empty.
	void release() noexcept {
		if (owns_ && ptr_ != nullptr) {
			free_(ptr_);
		}
		ptr_ = nullptr;
		len_ = 0;
		cap_ = 0;
		owns_ = false;
	}

	// Sets all bytes in the allocation to 0.
	void reset() {
		if (ptr_ == nullptr || cap_ == 0) return;
		std::memset(ptr_, 0, cap_ * sizeof(T));
	}

	const T* data() const noexcept { return ptr_; }
	T* data() noexcept { return ptr_; }

	std::size_t len() const noexcept { return len_; }
	std::size_t capacity() const noexcept { return cap_; }
	bool empty() const noexcept { return len_ == 0; }
	bool is_null() const noexcept { return ptr_ == nullptr; }

	// # Safety: caller guarantees new_len <= capacity().
	void set_len(std::size_t new_len) noexcept {
		assert(new_len <= cap_);
		len_ = new_len;
	}

	// # Safety: caller guarantees the whole allocation is initialized when used.
	void set_max_len() noexcept { len_ = cap_; }

	T& operator[](std::size_t idx) noexcept {
		assert(idx < len_);
		return ptr_[idx];
	}
	const T& operator[](std::size_t idx) const noexcept {
		assert(idx < len_);
		return ptr_[idx];
	}

private:
	static T* allocate_(std::size_t count) {
		void* p = std::malloc(count * sizeof(T));
		if (p == nullptr && count != 0) {
			std::fprintf(stderr, "HostBuffer malloc failed\n");
			std::exit(1);
		}
		return static_cast<T*>(p);
	}

	static void free_(T* ptr) noexcept { std::free(ptr); }

	void move_from_(HostBuffer& other) noexcept {
		ptr_ = other.ptr_;
		len_ = other.len_;
		cap_ = other.cap_;
		owns_ = other.owns_;

		other.ptr_ = nullptr;
		other.len_ = 0;
		other.cap_ = 0;
		other.owns_ = false;
	}

	T* ptr_{nullptr};
	std::size_t len_{0};
	std::size_t cap_{0};
	bool owns_{false};
};
