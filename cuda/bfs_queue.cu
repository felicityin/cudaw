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
    int* buffer1 = (int*)malloc(graph.num_vertices * sizeof(int));
    int* buffer2 = (int*)malloc(graph.num_vertices * sizeof(int));
    int* prev_frontier = buffer1;
    int* curr_frontier = buffer2;
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

    free(buffer1);
    free(buffer2);
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
            //  atomicCAS(&level[neighbor], INF, curr_level):
            //      if (level[neighbor] == INF) {
            //          level[neighbor] = curr_level;
            //      }
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
            //  atomicCAS(&level[neighbor], INF, curr_level):
            //      if (level[neighbor] == INF) {
            //          level[neighbor] = curr_level;
            //      }
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
    int src_vertex = 0;
    level[src_vertex] = 0;

    int* new_vertex_visited_d;
    CUDA_OK(cudaMalloc((void**)&new_vertex_visited_d, sizeof(int)));
    int* level_d;
    CUDA_OK(cudaMalloc((void**)&level_d, graph_d.num_vertices * sizeof(int)));
    CUDA_OK(cudaMemcpy(level_d, level, graph_d.num_vertices * sizeof(int), cudaMemcpyHostToDevice));

    int* buffer1_d;
    int* buffer2_d;
    CUDA_OK(cudaMalloc((void**)&buffer1_d, graph_d.num_vertices * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&buffer2_d, graph_d.num_vertices * sizeof(int)));
    int* prev_frontier_d = buffer1_d;
    int* curr_frontier_d = buffer2_d;
    CUDA_OK(cudaMemcpy(prev_frontier_d, &src_vertex, sizeof(int), cudaMemcpyHostToDevice));
    int* num_curr_frontier_d = buffer1_d + graph_d.num_vertices;
    CUDA_OK(cudaMalloc((void**)&num_curr_frontier_d, sizeof(int)));

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_OK(cudaEventCreate(&startEvent));
    CUDA_OK(cudaEventCreate(&stopEvent));

    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs_top_down(expected, graph);

    unsigned int numThreadsPerBlock = 256;
    int num_prev_frontier = 1;

    //--------------- BFS global queue -----------------
    CUDA_OK(cudaEventRecord(startEvent));

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        // Visit vertex in the previous frontier
        CUDA_OK(cudaMemset(num_curr_frontier_d, 0, sizeof(int)));
        unsigned int numBlocks = (num_prev_frontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_global_queue<<<numBlocks, numThreadsPerBlock>>>(level_d, prev_frontier_d, curr_frontier_d,
                                                            num_prev_frontier, num_curr_frontier_d, 
                                                            graph_d, curr_level);

        // Swap buffers
        int* tmp = prev_frontier_d;
        prev_frontier_d = curr_frontier_d;
        curr_frontier_d = tmp;
        CUDA_OK(cudaMemcpy(&num_prev_frontier, num_curr_frontier_d, sizeof(int), cudaMemcpyDeviceToHost));
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    float ms;
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[BFS global queue] Time: %f ms\n", ms);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(level, level_d, graph_d.num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(level, expected, n));

    //--------------- BFS private queue -----------------
    CUDA_OK(cudaEventRecord(startEvent));

    num_prev_frontier = 1;

    for (int curr_level = 1; num_prev_frontier > 0; ++curr_level) {
        // Visit vertex in the previous frontier
        CUDA_OK(cudaMemset(num_curr_frontier_d, 0, sizeof(int)));
        unsigned int numBlocks = (num_prev_frontier + numThreadsPerBlock - 1) / numThreadsPerBlock;
        bfs_priv_queue<<<numBlocks, numThreadsPerBlock>>>(level_d, prev_frontier_d, curr_frontier_d,
                                                            num_prev_frontier, num_curr_frontier_d, 
                                                            graph_d, curr_level);

        // Swap buffers
        int* tmp = prev_frontier_d;
        prev_frontier_d = curr_frontier_d;
        curr_frontier_d = tmp;
        CUDA_OK(cudaMemcpy(&num_prev_frontier, num_curr_frontier_d, sizeof(int), cudaMemcpyDeviceToHost));
    }

    CUDA_OK(cudaEventRecord(stopEvent));
    CUDA_OK(cudaEventSynchronize(stopEvent));
    CUDA_OK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    printf("[BFS private queue] Time: %f ms\n", ms);

    // Copy output to CPU
    CUDA_OK(cudaMemcpy(level, level_d, graph_d.num_vertices * sizeof(int), cudaMemcpyDeviceToHost));

    // Verify result
    assert(verify_result(level, expected, n));

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
