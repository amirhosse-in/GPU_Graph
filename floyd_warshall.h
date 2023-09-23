#ifndef BENCHMARK_FLOYD_WARSHALL_H
#define BENCHMARK_FLOYD_WARSHALL_H

/**
 * @brief Detects negative cycles in the final distance matrix.
 * 
 * This function checks if there is a negative cycle in the final distance matrix
 * by iterating over the diagonal elements of the matrix and checking if any of them
 * are negative. If a negative diagonal element is found, it means that there is a
 * negative cycle in the graph.
 * 
 * @param final_disance_matrix The final distance matrix after running the Floyd-Warshall algorithm.
 * @param vertices_count The number of vertices in the graph.
 */
void NegativeCycleDetector(int* final_disance_matrix, int vertices_count);

/**
 * @brief Computes all pairs shortest paths using the Floyd-Warshall algorithm on a GPU.
 * 
 * @param distance_matrix Pointer to the distance matrix in row-major order (input and output).
 * @param vertices_count Number of vertices in the graph.
 * @param edges_count Unused in this function (kept for consistency with your original code).
 */
void FloydWarshall(int* distance_matrix, int vertices_count);

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
void InitDistanceMatrix(int* &distance_matrix, int vertices_count, int edges_count, int* u, int* v, int* w);

#endif // BENCHMARK_FLOYD_WARSHALL_H