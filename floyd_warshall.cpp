#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "floyd_warshall.h"
#include "graph_io.h"

#define INF 999999999

using namespace std;

/**
 * @brief CUDA kernel for the Floyd-Warshall algorithm.
 * 
 * @param distance_matrix Pointer to the distance matrix in row-major order.
 * @param vertices_count Number of vertices in the graph.
 * @param edges_count Number of edges in the graph.
 * @param k Intermediate vertex index for the algorithm.
 */
// This function implements the Floyd-Warshall algorithm on a GPU using CUDA.
// It takes in a distance matrix, the number of vertices in the graph, the number of edges in the graph, and the current iteration k.
__global__ void FloydWarshallKernel(int* distance_matrix, int vertices_count, int k) {
    // Calculate the thread ID based on the block and thread indices.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the indices i and j based on the thread ID.
    int i = tid / vertices_count;
    int j = tid % vertices_count;

    // Check if the thread ID is within the range of valid edges.
    if (tid < vertices_count * vertices_count) {
        // Retrieve the distances between i and j, i and k, and k and j.
        int ij = distance_matrix[tid];
        int ik = distance_matrix[i * vertices_count + k];
        int kj = distance_matrix[k * vertices_count + j];

        // Update the distance between i and j to the minimum of the current distance and the sum of the distances between i and k and k and j.
        distance_matrix[tid] = min(ij, ik + kj);
    }
}


void NegativeCycleDetector(int* final_disance_matrix, int vertices_count) {
    // Check if there is a negative cycle in the graph
    for(int i = 0; i < vertices_count; i++){
        if (final_disance_matrix[i * vertices_count + i] < 0) {
            cout << "Negative cycle detected!" << endl;
            return;
        }
    }
}


void FloydWarshall(int* distance_matrix, int vertices_count) {
    // Calculate the total number of elements in the distance matrix.
    const int squaredVertices = vertices_count * vertices_count;

    // Allocate memory for the distance matrix on the GPU.
    int* d_distance_matrix;
    cudaMalloc(&d_distance_matrix, squaredVertices * sizeof(int));

    // Copy the initial distance matrix from host to device.
    cudaMemcpy(d_distance_matrix, distance_matrix, squaredVertices * sizeof(int), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions.
    dim3 threadsPerBlock(1024);
    dim3 numBlocks((squaredVertices + 1023) / 1024);

    // Perform the Floyd-Warshall algorithm for all vertices.
    for (int k = 0; k < vertices_count; k++) {
        FloydWarshallKernel<<<numBlocks, threadsPerBlock>>>(d_distance_matrix, vertices_count, k);
    }

    // Copy the updated distance matrix from device to host.
    cudaMemcpy(distance_matrix, d_distance_matrix, squaredVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Check for negative cycles.
    NegativeCycleDetector(distance_matrix, vertices_count);

    // Free memory on the GPU.
    cudaFree(d_distance_matrix);
}

void InitDistanceMatrix(int* &distance_matrix, int vertices_count, int edges_count, int* u, int* v, int* w) {
    // Allocate memory for the distance matrix
    distance_matrix = new int[vertices_count * vertices_count];

    // Initialize the distance matrix
    for(int i = 0; i < vertices_count * vertices_count; i++){
        // If the current index is on the diagonal, set the distance to 0
        if (i % vertices_count == i / vertices_count) {
            distance_matrix[i] = 0;
        } else {
            // Otherwise, set the distance to infinity
            distance_matrix[i] = INF;
        }
    }

    // Set the distances for the edges
    for(int i = 0; i < edges_count; i++){
        distance_matrix[u[i] * vertices_count + v[i]] = w[i];
    }
}

