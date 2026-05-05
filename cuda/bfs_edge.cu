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
struct COOGraph {
    int* src;
    int* dst;
    int num_vertices, num_edges;
};

void cpu_bfs(int* level, const COOGraph<int> graph) {
    int curr_level = 1;
    int new_vertex_visited = 1;

    for (int vertex = 0; vertex < graph.num_vertices; vertex++) {
        new_vertex_visited = 0;

        for (int edge = 0; edge < graph.num_edges; edge++) {
            int src = graph.src[edge];
            int dst = graph.dst[edge];
            // Source vertex is in the previous level and destination vertex is not visited
            if (level[src] == curr_level - 1 && level[dst] == INF) {
                // Mark destination vertex as part of current level
                level[dst] = curr_level;
                new_vertex_visited = 1;
            }
        }

        if (!new_vertex_visited) {
            break;
        }

        curr_level++;
    }
}

__global__ void bfs(int* level, int* new_vertex_visited, const COOGraph<int> graph, int curr_level) { 
    // Assign a thread to every edge
    unsigned int edge = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge < graph.num_edges) {
        int src = graph.src[edge];
        int dst = graph.dst[edge];
        // Source vertex is in the previous level and destination vertex is not visited
        if (level[src] == curr_level - 1 && level[dst] == INF) {
            // Mark destination vertex as part of current level
            level[dst] = curr_level;
            *new_vertex_visited = 1;
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

    COOGraph<int> graph;
    graph.num_vertices = n;
    graph.num_edges = edges.size();
    graph.src = (int*)malloc(edges.size() * sizeof(int));
    graph.dst = (int*)malloc(edges.size() * sizeof(int));
    
    for (int i = 0; i < edges.size(); i++) {
        graph.src[i] = edges[i].first;
        graph.dst[i] = edges[i].second;
    }
    // -------------- Construct CSRGraph end ---------------

    COOGraph<int> graph_d;
    graph_d.num_vertices = graph.num_vertices;
    graph_d.num_edges = graph.num_edges;
    CUDA_OK(cudaMalloc((void**)&graph_d.src, graph_d.num_edges * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&graph_d.dst, graph_d.num_edges * sizeof(int)));
    CUDA_OK(cudaMemcpy(graph_d.src, graph.src, graph_d.num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(graph_d.dst, graph.dst, graph_d.num_edges * sizeof(int), cudaMemcpyHostToDevice));

    int* expected = (int*)malloc(n * sizeof(int));
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs(expected, graph);

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
    unsigned int numBlocks = (graph_d.num_edges + numThreadsPerBlock - 1) / numThreadsPerBlock;
    int new_vertex_visited = 1;

    CUDA_OK(cudaEventRecord(startEvent));

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
        new_vertex_visited = 0;
        CUDA_OK(cudaMemcpy(new_vertex_visited_d, &new_vertex_visited, sizeof(int), cudaMemcpyHostToDevice));

        bfs<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);

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

    printf("BFS/CSR completed successfully.\n");

    free(graph.src);
    free(graph.dst);
    free(level);
    free(expected);
    CUDA_OK(cudaFree(graph_d.src));
    CUDA_OK(cudaFree(graph_d.dst));
    CUDA_OK(cudaFree(new_vertex_visited_d));
    CUDA_OK(cudaFree(level_d));
    CUDA_OK(cudaEventDestroy(startEvent));
    CUDA_OK(cudaEventDestroy(stopEvent));;
    return 0;
}
