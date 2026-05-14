#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <vector>

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
    int new_vertex_visited = 1;

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;

        for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
            // Check if the vertex is in the previous level
            if (level[vertex] == curr_level - 1) {
                for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
                    int neighbor = graph.col_indices[edge];
                    if (level[neighbor] == INF) {
                        // Add the neighbor to the current level
                        level[neighbor] = curr_level;
                        // Tell the host we're gonna have to do another iteration because we visited a new vertex
                        new_vertex_visited = 1;
                    }
                }
            }
        }
    }
}

void cpu_bfs_bottom_up(int* level, const CSRGraph<int> graph) {
    int new_vertex_visited = 1;

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;

        for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
            for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
                int neighbor = graph.col_indices[edge];
                // Check if any of neighbors are in the previous level
                if (level[neighbor] == curr_level - 1) {
                    // Mark myself as being part of the current level
                    level[vertex] = curr_level;
                    // Tell the host we're gonna have to do another iteration because we visited a new vertex
                    new_vertex_visited = 1;
                    break;
                }
            }
        }
    }
}

__global__ void bfs_top_down(int* level, int* new_vertex_visited, const CSRGraph<int> graph, int curr_level) { 
    // Assign a thread to every vertex
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < graph.num_vertices) {
        // Check if the vertex is in the previous level
        if (level[vertex] == curr_level - 1) {
            for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
                int neighbor = graph.col_indices[edge];
                if (level[neighbor] == INF) {
                    // Add the neighbor to the current level
                    level[neighbor] = curr_level;
                    // Tell the host we're gonna have to do another iteration because we visited a new vertex
                    *new_vertex_visited = 1;
                }
            }
        }
    }
}

__global__ void bfs_bottom_up(int* level, int* new_vertex_visited, const CSRGraph<int> graph, int curr_level) { 
    // Assign a thread to every vertex
    unsigned int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if (vertex < graph.num_vertices) {
        if (level[vertex] == INF) {
            for (int edge = graph.row_ptrs[vertex]; edge < graph.row_ptrs[vertex + 1]; edge++) {
                int neighbor = graph.col_indices[edge];
                // Check if any of neighbors are in the previous level
                if (level[neighbor] == curr_level - 1) {
                    // Mark myself as being part of the current level
                    level[vertex] = curr_level;
                    // Tell the host we're gonna have to do another iteration because we visited a new vertex
                    *new_vertex_visited = 1;
                    break;
                }
            }
        }
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
    CUDA_OK(cudaMemcpy(graph_d.row_ptrs, graph.row_ptrs, (graph_d.num_vertices + 1) * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(graph_d.col_indices, graph.col_indices, graph_d.num_edges * sizeof(int),
                       cudaMemcpyHostToDevice));

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    auto level_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* expected = expected_buf.data();
    int* level = level_buf.data();
    memset(level, INF, n * sizeof(int));
    level[0] = 0;

    auto new_vertex_visited_d_buf = CudaBuffer<int>::with_capacity(1);
    auto level_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_vertices));
    int* new_vertex_visited_d = new_vertex_visited_d_buf.data();
    int* level_d = level_d_buf.data();
    CUDA_OK(cudaMemcpy(level_d, level, graph_d.num_vertices * sizeof(int), cudaMemcpyHostToDevice));

    CudaTimer timer;
    float ms = 0.0f;

    unsigned int numThreadsPerBlock = 128;
    unsigned int numBlocks = (graph_d.num_vertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
    int new_vertex_visited = 1;

    //--------------- BFS top down -----------------
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_top_down(expected, graph);

    timer.start();

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        bfs_top_down<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);

        CUDA_OK(cudaMemcpy(&new_vertex_visited, new_vertex_visited_d, sizeof(int), cudaMemcpyDeviceToHost));
    }

    ms = timer.elapsed_ms();
    printf("[top down] Time: %f ms\n", ms);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(level, level_d, graph_d.num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(level, expected, n));

    //--------------- BFS bottom up -----------------
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_bottom_up(expected, graph);

    timer.start();

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        bfs_bottom_up<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);

        CUDA_OK(cudaMemcpy(&new_vertex_visited, new_vertex_visited_d, sizeof(int), cudaMemcpyDeviceToHost));
    }

    ms = timer.elapsed_ms();
    printf("[bottom up] Time: %f ms\n", ms);

    //--------------- BFS direction optimizied -----------------
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_bottom_up(expected, graph);

    timer.start();

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        if (curr_level == 1) {
            bfs_top_down<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);
        } else {
            bfs_bottom_up<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);
        }

        CUDA_OK(cudaMemcpy(&new_vertex_visited, new_vertex_visited_d, sizeof(int), cudaMemcpyDeviceToHost));
    }

    ms = timer.elapsed_ms();
    printf("[direction optimizied] Time: %f ms\n", ms);

    printf("BFS/CSR completed successfully.\n");
    return 0;
}
