%%writefile sample.cu
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define INF 1000000

using namespace std;


/**
 * Reads a graph from a file and stores its edges in three arrays: u, v, and w.
 * 
 * @param filename The name of the file to read the graph from.
 * @param u A pointer to an integer array that will store the source vertices of the edges.
 * @param v A pointer to an integer array that will store the destination vertices of the edges.
 * @param w A pointer to an integer array that will store the weights of the edges.
 * @param edges_count A reference to an integer that will store the number of edges in the graph.
 * @param vertices_count A reference to an integer that will store the number of vertices in the graph.
 */
void ReadGraphFromFile(const char *filename, vector<int> &u, vector<int> &v, vector<int> &w, int &edges_count, int &vertices_count)
{
    
    // Open the input file
    std::ifstream file(filename);
    // Declare a string variable to store each line of the file
    string line;
    
    // Check if the file was opened successfully
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    
    // Declare variables to store the source vertex, destination vertex, and weight of each edge
    int source, dest, weight;
    // Declare a character variable to store the first character of each line
    char c;

    // Declare a counter variable to keep track of the number of edges read
    int i = 0;
    // Read each line of the file
    while (std::getline(file, line)) {
        // Create a stringstream object from the line
        stringstream ss(line);

        // Read the first character of the line
        ss >> c;
        // If the first character is not 'a', return from the function (not a edge line)
        if (c != 'a')
            continue;

        // Read the source vertex, destination vertex, and weight of the edge
        ss >> source >> dest >> weight;
        // Push the source vertex, destination vertex, and weight of the edge to the corresponding arrays
        u.push_back(source - 1); // Subtract 1 from the source and destination vertices to convert them to 0-based indexing
        v.push_back(dest - 1);
        w.push_back(weight);
        // Increment the counter variable
        i++;   
    }

    // Store the number of edges read in the edges_count variable
    edges_count = i;
    // Store the number of vertices in the graph in the vertices_count variable
    vertices_count = max(*max_element(u.begin(), u.end()), *max_element(v.begin(), v.end())) + 1;
    
    // Close the input file
    file.close();
}



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
__global__ void floyd_warshall_kernel(int* distance_matrix, int vertices_count, int edges_count, int k) {
    // Calculate the thread ID based on the block and thread indices.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the indices i and j based on the thread ID.
    int i = tid / vertices_count;
    int j = tid % vertices_count;

    // Check if the thread ID is within the range of valid edges and if i, j, and k are not equal.
    if (tid < edges_count && i != j && i != k && j != k) {
        // Retrieve the distances between i and j, i and k, and k and j.
        int ij = distance_matrix[tid];
        int ik = distance_matrix[i * vertices_count + k];
        int kj = distance_matrix[k * vertices_count + j];

        // Update the distance between i and j to the minimum of the current distance and the sum of the distances between i and k and k and j.
        distance_matrix[tid] = min(ij, ik + kj);
    }
}

/**
 * @brief Computes all pairs shortest paths using the Floyd-Warshall algorithm on a GPU.
 * 
 * @param distance_matrix Pointer to the distance matrix in row-major order (input and output).
 * @param vertices_count Number of vertices in the graph.
 * @param edges_count Unused in this function (kept for consistency with your original code).
 */
void floyd_warshall(int* distance_matrix, int vertices_count, int edges_count) {
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
      floyd_warshall_kernel<<<numBlocks, threadsPerBlock>>>(d_distance_matrix, vertices_count, edges_count, k);
    }

    // Copy the updated distance matrix from device to host.
    cudaMemcpy(distance_matrix, d_distance_matrix, squaredVertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Free memory on the GPU.
    cudaFree(d_distance_matrix);
}

/**
 * @brief the distance matrix for the Floyd-Warshall algorithm.
 * 
 * @param dist Pointer to the distance matrix to be initialized.
 * @param vertices_count Number of vertices in the graph.
 * @param edges_count Number of edges in the graph.
 * @param u Array of size edges_count containing the source vertices of the edges.
 * @param v Array of size edges_count containing the destination vertices of the edges.
 * @param w Array of size edges_count containing the weights of the edges.
 */
void init_distance_matrix(int* &distance_matrix, int vertices_count, int edges_count, int* u, int* v, int* w) {
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

int main() {
    vector<int> u, v, w;
    int edges_count, vertices_count;
    ReadGraphFromFile("small graph.gr", u, v, w, edges_count, vertices_count);
    int* distance_matrix;
    init_distance_matrix(distance_matrix, vertices_count, edges_count, u.data(), v.data(), w.data());
    floyd_warshall(distance_matrix, vertices_count, edges_count);
    int value;
    for(int i = 0; i < vertices_count; i++){
        for(int j = 0; j < vertices_count; j++){
            value = distance_matrix[i * vertices_count + j];
            cout << ((value == 1000000) ? "INF\t" : to_string(value) + "\t");
        }
        cout << endl;
    }
    return 0;
}
