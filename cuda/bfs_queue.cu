#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <vector>
#include <algorithm>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int INF = -1;

template<typename T>
struct CSRGraph {
    int* row_ptrs;
    T* col_indices;
    int num_vertices, num_edges;
};

void cpu_bfs_top_down(int* level, const CSRGraph<int> graph) {
    auto buffer1_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(graph.num_vertices));
    auto buffer2_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(graph.num_vertices));
    int* prev_frontier = buffer1_buf.data();
    int* curr_frontier = buffer2_buf.data();
    prev_frontier[0] = 0;
    int num_prev_frontier = 1;
    int num_curr_frontier = 0;

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        num_curr_frontier = 0;

        for (int i = 0; i < num_prev_frontier; ++i) {
            int vertex = prev_frontier[i];

            for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
                int neighbor = graph.col_indices[edge];
                // If the neighbor has not been visited yet, add it to the current frontier
                if (level[neighbor] == INF) {
                    level[neighbor] = curr_level;
                    curr_frontier[num_curr_frontier++] = neighbor;
                }
            }
        }

        // Swap buffers
        int* tmp = prev_frontier;
        prev_frontier = curr_frontier;
        curr_frontier = tmp;

        num_prev_frontier = num_curr_frontier;
    }
}

__global__ void bfs_global_queue(int* level, int* prev_frontier, int* curr_frontier,
                                 int num_prev_frontier, int* num_curr_frontier,
                                 const CSRGraph<int> graph, int curr_level) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_prev_frontier) {
        // Every thread processes one vertex in the previous frontier
        int vertex = prev_frontier[i];
        for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
            int neighbor = graph.col_indices[edge];
            // If the neighbor has not been visited yet, add it to the current frontier
            if (atomicCAS(&level[neighbor], INF, curr_level) == INF) {
                int curr_frontier_index = atomicAdd(num_curr_frontier, 1); // return the old value
                curr_frontier[curr_frontier_index] = neighbor;
            }
        }
    }
}

#define LOCAL_QUEUE_SIZE 2048

__global__ void bfs_priv_queue(int* level, int* prev_frontier, int* curr_frontier,
                               int num_prev_frontier, int* num_curr_frontier,
                               const CSRGraph<int> graph, int curr_level) { 
    __shared__ int curr_frontier_s[LOCAL_QUEUE_SIZE];
    __shared__ int num_curr_frontier_s;
    if (threadIdx.x == 0) {
        num_curr_frontier_s = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_prev_frontier) {
        int vertex = prev_frontier[i];
        for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
            int neighbor = graph.col_indices[edge];
            // If the neighbor has not been visited yet, add it to the current frontier
            if (atomicCAS(&level[neighbor], INF, curr_level) == INF) {
                int curr_frontier_index = atomicAdd(&num_curr_frontier_s, 1); // return the old value
                if (curr_frontier_index < LOCAL_QUEUE_SIZE) {
                    // Not overflow: push to local queue
                    curr_frontier_s[curr_frontier_index] = neighbor;
                } else {
                    // Overflow: push to global queue
                    num_curr_frontier_s = LOCAL_QUEUE_SIZE;
                    int curr_frontier_index = atomicAdd(num_curr_frontier, 1); // return the old value
                    curr_frontier[curr_frontier_index] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // All the thread wait for thread zero to allocate space in the global queue
    __shared__ int curr_frontier_start_idx;
    if (threadIdx.x == 0) {
        curr_frontier_start_idx = atomicAdd(num_curr_frontier, num_curr_frontier_s); // return the old value
    }
    __syncthreads();

    // Copy the current frontier to the global queue
    if (threadIdx.x < num_curr_frontier_s) {
        for (int i = threadIdx.x; i < num_curr_frontier_s; i += blockDim.x) {
            curr_frontier[curr_frontier_start_idx + i] = curr_frontier_s[i];
        }
    }
}

__global__ void bfs_child(int* level, int* prev_frontier, int* curr_frontier,
                          int num_prev_frontier, int* num_curr_frontier,
                          const CSRGraph<int> graph, int curr_level,
                          int num_neighbors, int start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_neighbors) {
        int edge = start + i;
        int neighbor = graph.col_indices[edge];
        if (atomicCAS(&level[neighbor], INF, curr_level) == INF) {
            int curr_frontier_index = atomicAdd(num_curr_frontier, 1); // return the old value
            curr_frontier[curr_frontier_index] = neighbor;
        }
    }
}

__global__ void bfs_dynamic_parallel(int* level, int* prev_frontier, int* curr_frontier,
                                     int num_prev_frontier, int* num_curr_frontier,
                                     const CSRGraph<int> graph, int curr_level) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_prev_frontier) {
        // Every thread processes one vertex in the previous frontier
        int vertex = prev_frontier[i];
        int start = graph.row_ptrs[vertex];
        int num_neighbors = graph.row_ptrs[vertex + 1] - start;

        int num_threads_per_block = 64;
        int num_blocks = (num_neighbors + num_threads_per_block - 1) / num_threads_per_block;

        bfs_child<<<num_blocks, num_threads_per_block>>>(level, prev_frontier, curr_frontier,
                                                         num_prev_frontier, num_curr_frontier,
                                                         graph, curr_level,
                                                         num_neighbors, start);
    }
}

__global__ void bfs_offload_driver(int* level, int* prev_frontier, int* curr_frontier,
                                   int* num_curr_frontier, const CSRGraph<int> graph) { 
    int numThreadsPerBlock = 256;
    int num_prev_frontier = 1;

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        // Visit vertex in the previous frontier
        *num_curr_frontier = 0;
        unsigned int numBlocks = (num_prev_frontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_priv_queue<<<numBlocks, numThreadsPerBlock>>>(level, prev_frontier, curr_frontier,
                                                          num_prev_frontier, num_curr_frontier, 
                                                          graph, curr_level);

        // Swap buffers
        int* tmp = prev_frontier;
        prev_frontier = curr_frontier;
        curr_frontier = tmp;

        num_prev_frontier = *num_curr_frontier;
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

// nvcc -rdc=true --default-stream per-thread bfs_queue.cu
int main() {
    // -------------- Construct CSRGraph start ---------------
    int n = 5;
    std::vector<std::pair<int, int>> edges = {
        {0, 1}, {0, 2}, {0, 3}, {0, 4},
        {1, 2}, {1, 3}, {1, 4},
        {2, 3}, {2, 4},
        {3, 4},
    };

    CSRGraph<int> graph;
    graph.num_vertices = n;
    graph.num_edges = edges.size();
    auto row_ptrs_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n + 1));
    auto col_indices_buf = HostBuffer<int>::with_capacity(edges.size());
    graph.row_ptrs = row_ptrs_buf.data();
    graph.col_indices = col_indices_buf.data();

    std::vector<int> degree(n, 0);
    for (const auto& [u, v] : edges) {
        if (u >= 0 && u < n && v >= 0 && v < n) {
            degree[u]++;
        }
    }

    graph.row_ptrs[0] = 0;
    for (int i = 0; i < n; ++i) {
        graph.row_ptrs[i + 1] = graph.row_ptrs[i] + degree[i];
    }

    std::vector<int> fill_pos(n, 0);
    for (size_t i = 0; i < edges.size(); ++i) {
        int u = edges[i].first;
        int v = edges[i].second;
        int pos = graph.row_ptrs[u] + fill_pos[u];
        graph.col_indices[pos] = v;
        fill_pos[u]++;
    }
    // -------------- Construct CSRGraph end ---------------

    CSRGraph<int> graph_d;
    graph_d.num_vertices = graph.num_vertices;
    graph_d.num_edges = graph.num_edges;
    auto row_ptrs_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_vertices + 1));
    auto col_indices_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_edges));
    graph_d.row_ptrs = row_ptrs_d_buf.data();
    graph_d.col_indices = col_indices_d_buf.data();
    row_ptrs_d_buf.copy_from_host(graph.row_ptrs, static_cast<std::size_t>(graph_d.num_vertices + 1));
    col_indices_d_buf.copy_from_host(graph.col_indices, static_cast<std::size_t>(graph_d.num_edges));

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto level_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* expected = expected_buf.data();
    int* level = level_buf.data();
    memset(level, INF, n * sizeof(int));
    int src_vertex = 0;
    level[src_vertex] = 0;

    auto level_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_vertices));
    int* level_d = level_d_buf.data();

    auto buffer1_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_vertices));
    auto buffer2_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_vertices));
    int* prev_frontier_d = buffer1_d_buf.data();
    int* curr_frontier_d = buffer2_d_buf.data();

    auto num_curr_frontier_d_buf = CudaBuffer<int>::with_capacity(1);
    int* num_curr_frontier_d = num_curr_frontier_d_buf.data();

    auto reset_state = [&]() {
        memset(level, INF, n * sizeof(int));
        level[src_vertex] = 0;
        level_d_buf.copy_from_host(level, static_cast<std::size_t>(graph_d.num_vertices));
        auto prev_frontier_tmp = CudaBuffer<int>::from_raw_parts(prev_frontier_d, 1, 1);
        prev_frontier_tmp.copy_from_host(&src_vertex, 1);
    };

    CudaTimer timer;
    float ms = 0.0f;

    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_top_down(expected, graph);

    unsigned int numThreadsPerBlock = 256;
    int num_prev_frontier = 1;

    //--------------- BFS global queue -----------------
    reset_state();
    num_prev_frontier = 1;
    timer.start();

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        // Visit vertex in the previous frontier
		num_curr_frontier_d_buf.fill_zero();
        unsigned int numBlocks = (num_prev_frontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_global_queue<<<numBlocks, numThreadsPerBlock>>>(level_d, prev_frontier_d, curr_frontier_d,
                                                            num_prev_frontier, num_curr_frontier_d, 
                                                            graph_d, curr_level);

        // Swap buffers
        int* tmp = prev_frontier_d;
        prev_frontier_d = curr_frontier_d;
        curr_frontier_d = tmp;

        num_curr_frontier_d_buf.copy_to_host(&num_prev_frontier, 1);
		num_curr_frontier_d_buf.synchronize();
    }
    ms = timer.elapsed_ms();
    printf("[BFS global queue] Time: %f ms\n", ms);

    // Copy output to CPU
    level_d_buf.copy_to_host(level, static_cast<std::size_t>(graph_d.num_vertices));
	level_d_buf.synchronize();

    // Verify result
    assert(verify_result(level, expected, n));

    //--------------- BFS private queue -----------------
    reset_state();
    num_prev_frontier = 1;
    timer.start();

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        // Visit vertex in the previous frontier
		num_curr_frontier_d_buf.fill_zero();
        unsigned int numBlocks = (num_prev_frontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_priv_queue<<<numBlocks, numThreadsPerBlock>>>(level_d, prev_frontier_d, curr_frontier_d,
                                                            num_prev_frontier, num_curr_frontier_d, 
                                                            graph_d, curr_level);

        // Swap buffers
        int* tmp = prev_frontier_d;
        prev_frontier_d = curr_frontier_d;
        curr_frontier_d = tmp;
        num_curr_frontier_d_buf.copy_to_host(&num_prev_frontier, 1);
		num_curr_frontier_d_buf.synchronize();
    }
    ms = timer.elapsed_ms();
    printf("[BFS private queue] Time: %f ms\n", ms);

    // Copy output to CPU
    level_d_buf.copy_to_host(level, static_cast<std::size_t>(graph_d.num_vertices));
    level_d_buf.synchronize();

    // Verify result
    assert(verify_result(level, expected, n));

    //--------------- BFS dynamic parellel -----------------
    // We should increase the pending launch count to match the number of vertices
    CUDA_OK(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, graph_d.num_vertices));
    reset_state();
    num_prev_frontier = 1;
    timer.start();

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        // Visit vertex in the previous frontier
        num_curr_frontier_d_buf.fill_zero();
        unsigned int numBlocks = (num_prev_frontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_dynamic_parallel<<<numBlocks, numThreadsPerBlock>>>(level_d, prev_frontier_d, curr_frontier_d,
                                                                num_prev_frontier, num_curr_frontier_d, 
                                                                graph_d, curr_level);

        // Swap buffers
        int* tmp = prev_frontier_d;
        prev_frontier_d = curr_frontier_d;
        curr_frontier_d = tmp;
        num_curr_frontier_d_buf.copy_to_host(&num_prev_frontier, 1);
        num_curr_frontier_d_buf.synchronize();
    }
    ms = timer.elapsed_ms();
    printf("[BFS dynamic parellel] Time: %f ms\n", ms);

     // Copy output to CPU
    level_d_buf.copy_to_host(level, static_cast<std::size_t>(graph_d.num_vertices));
    level_d_buf.synchronize();

    // Verify result
    assert(verify_result(level, expected, n));

    //--------------- BFS offload driver code -----------------
    reset_state();
    timer.start();

    bfs_offload_driver<<<1, 1>>>(level_d, prev_frontier_d, curr_frontier_d, num_curr_frontier_d, graph_d);
    ms = timer.elapsed_ms();
    printf("[BFS offload driver code] Time: %f ms\n", ms);

    // Copy output to CPU
    level_d_buf.copy_to_host(level, static_cast<std::size_t>(graph_d.num_vertices));
    level_d_buf.synchronize();

    // Verify result
    assert(verify_result(level, expected, n));

    printf("BFS/CSR completed successfully.\n");
    return 0;
}
