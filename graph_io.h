#ifndef BENCHMARK_GRAPH_IO_H
#define BENCHMARK_GRAPH_IO_H

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
void ReadGraphFromFile(const char *filename, vector<int> &u, vector<int> &v, vector<int> &w, int &edges_count, int &vertices_count);

/**
 * Writes the result of the Floyd-Warshall algorithm to a file.
 * 
 * @param distance_matrix Pointer to the distance matrix containing the shortest paths between all pairs of vertices.
 * @param vertices_count The number of vertices in the graph.
 * @param filename The name of the file to write the result to.
 */
void WriteFloydWarshallResultToFile(const int* distance_matrix, const int vertices_count, const string& filename);

#endif // BENCHMARK_GRAPH_IO_H