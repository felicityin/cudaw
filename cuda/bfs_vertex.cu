#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

#define CUDA_OK(expr) \
    do { \
        cudaError_t code = expr; \
        if (code != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

const int INF = -1;

template<typename T>
struct CSRGraph {
    int* row_ptrs;
    T* col_indices;
    int num_vertices, num_edges;
};

void cpu_bfs_top_down(int* level, const CSRGraph<int> graph) {
    int curr_level = 1;
    int new_vertex_visited = 1;

    for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
        new_vertex_visited = 0;

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

        if (!new_vertex_visited) {
            break;
        }

        curr_level++;
    }
}

void cpu_bfs_bottom_up(int* level, const CSRGraph<int> graph) {
    int curr_level = 1;
    int new_vertex_visited = 1;

    for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
        new_vertex_visited = 0;

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

        if (!new_vertex_visited) {
            break;
        }

        curr_level++;
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
    graph.row_ptrs = (int*)malloc((n + 1) * sizeof(int));
    graph.col_indices = (int*)malloc(edges.size() * sizeof(int));
    
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
    CUDA_OK(cudaMalloc((void**)&graph_d.row_ptrs, (graph_d.num_vertices + 1) * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&graph_d.col_indices, graph_d.num_edges * sizeof(int)));
    CUDA_OK(cudaMemcpy(graph_d.row_ptrs, graph.row_ptrs, (graph_d.num_vertices + 1) * sizeof(int),
                       cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(graph_d.col_indices, graph.col_indices, graph_d.num_edges * sizeof(int),
                       cudaMemcpyHostToDevice));

    int* expected = (int*)malloc(n * sizeof(int));

    int* level = (int*)malloc(n * sizeof(int));
    memset(level, INF, n * sizeof(int));
    level[0] = 0;

    int* new_vertex_visited_d;
    CUDA_OK(cudaMalloc(&new_vertex_visited_d, sizeof(int)));
    int* level_d;
    CUDA_OK(cudaMalloc(&level_d, graph_d.num_vertices * sizeof(int)));
    CUDA_OK(cudaMemcpy(level_d, level, graph_d.num_vertices * sizeof(int), cudaMemcpyHostToDevice));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    unsigned int numThreadsPerBlock = 128;
    unsigned int numBlocks = (graph_d.num_vertices + numThreadsPerBlock - 1) / numThreadsPerBlock;
    int new_vertex_visited = 1;

    //--------------- BFS top down -----------------
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_top_down(expected, graph);

    CUDA_OK(cudaEventRecord(startEvent));

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        bfs_top_down<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);

        CUDA_OK(cudaMemcpy(&new_vertex_visited, new_vertex_visited_d, sizeof(int), cudaMemcpyDeviceToHost));
        if (!new_vertex_visited) {
            break;
        }
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[top down] Time: %f ms\n", ms);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(level, level_d, graph_d.num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(level, expected, n));

    //--------------- BFS bottom up -----------------
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_bottom_up(expected, graph);

    CUDA_OK(cudaEventRecord(startEvent));

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        bfs_bottom_up<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);

        CUDA_OK(cudaMemcpy(&new_vertex_visited, new_vertex_visited_d, sizeof(int), cudaMemcpyDeviceToHost));
        if (!new_vertex_visited) {
            break;
        }
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[bottom up] Time: %f ms\n", ms);

    //--------------- BFS direction optimizied -----------------
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_bottom_up(expected, graph);

    CUDA_OK(cudaEventRecord(startEvent));

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        if (curr_level == 1) {
            bfs_top_down<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);
        } else {
            bfs_bottom_up<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);
        }

        CUDA_OK(cudaMemcpy(&new_vertex_visited, new_vertex_visited_d, sizeof(int), cudaMemcpyDeviceToHost));
        if (!new_vertex_visited) {
            break;
        }
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[direction optimizied] Time: %f ms\n", ms);

    printf("BFS/CSR completed successfully.\n");

    free(graph.row_ptrs);
    free(graph.col_indices);
    free(level);
    free(expected);
    CUDA_OK(cudaFree(graph_d.row_ptrs));
    CUDA_OK(cudaFree(graph_d.col_indices));
    CUDA_OK(cudaFree(new_vertex_visited_d));
    CUDA_OK(cudaFree(level_d));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));;
    return 0;
}
