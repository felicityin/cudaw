#include <stdio.h>
#include <assert.h>
#include <cstring>
#include <vector>

#include "include/buffer.cuh"
#include "include/exception.cuh"
#include "include/timer.cuh"

const int INF = -1;

template<typename T>
struct COOGraph {
    int* src;
    int* dst;
    int num_vertices, num_edges;
};

void cpu_bfs(int* level, const COOGraph<int> graph) {
    int new_vertex_visited = 1;

    for (int curr_level = 1; new_vertex_visited; ++curr_level) {
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
    auto src_buf = HostBuffer<int>::with_capacity(edges.size());
    auto dst_buf = HostBuffer<int>::with_capacity(edges.size());
    graph.src = src_buf.data();
    graph.dst = dst_buf.data();

    for (int i = 0; i < edges.size(); i++) {
        graph.src[i] = edges[i].first;
        graph.dst[i] = edges[i].second;
    }
    // -------------- Construct CSRGraph end ---------------

    COOGraph<int> graph_d;
    graph_d.num_vertices = graph.num_vertices;
    graph_d.num_edges = graph.num_edges;
    auto src_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_edges));
    auto dst_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_edges));
    graph_d.src = src_d_buf.data();
    graph_d.dst = dst_d_buf.data();
	src_d_buf.copy_from_host(src_buf.data(), graph_d.num_edges);
	dst_d_buf.copy_from_host(dst_buf.data(), graph_d.num_edges);

    auto expected_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* expected = expected_buf.data();
    memset(expected, INF, n * sizeof(int));
    expected[0] = 0;
    cpu_bfs(expected, graph);

    auto level_buf = HostBuffer<int>::with_capacity(static_cast<std::size_t>(n));
    int* level = level_buf.data();
    memset(level, INF, n * sizeof(int));
    level[0] = 0;

    auto new_vertex_visited_d_buf = CudaBuffer<int>::with_capacity(1);
    auto level_d_buf = CudaBuffer<int>::with_capacity(static_cast<std::size_t>(graph_d.num_vertices));
    int* new_vertex_visited_d = new_vertex_visited_d_buf.data();
    int* level_d = level_d_buf.data();
	level_d_buf.copy_from_host(level, graph_d.num_vertices);

    CudaTimer timer;
    float ms = 0.0f;

    unsigned int numThreadsPerBlock = 128;
    unsigned int numBlocks = (graph_d.num_edges + numThreadsPerBlock - 1) / numThreadsPerBlock;
    int new_vertex_visited = 1;

    timer.start();

	for (int curr_level = 1; new_vertex_visited; ++curr_level) {
		new_vertex_visited = 0;
		new_vertex_visited_d_buf.copy_from_host(&new_vertex_visited, 1);

		bfs<<<numBlocks, numThreadsPerBlock>>>(level_d, new_vertex_visited_d, graph_d, curr_level);
		CUDA_OK(cudaGetLastError());

		new_vertex_visited_d_buf.copy_to_host(&new_vertex_visited, 1);
		new_vertex_visited_d_buf.synchronize();
	}

    ms = timer.elapsed_ms();
    printf("[top down] Time: %f ms\n", ms);

	// Copy output to CPU
	level_d_buf.copy_to_host(level, graph_d.num_vertices);
	level_d_buf.synchronize();

    // Verify result
    assert(verify_result(level, expected, n));

    printf("BFS/CSR completed successfully.\n");
    return 0;
}
