#ifndef BENCHMARK_GRAPH_IO_H
#define BENCHMARK_GRAPH_IO_H

#include <vector>
#include <fstream>
#include <sstream>

#endif // BENCHMARK_GRAPH_IO_H

using namespace std;



/**
 * This function reads a graph from a file and stores its edges and weights in three separate vectors.
 * The function takes the filename as input and three vectors of integers as references to store the source vertices of the edges, 
 * the destination vertices of the edges, and the weights of the edges.
 * 
 * @param filename The name of the file to read the graph from.
 * @param U A reference to a vector of integers to store the source vertices of the edges.
 * @param V A reference to a vector of integers to store the destination vertices of the edges.
 * @param W A reference to a vector of integers to store the weights of the edges.
 */
void read_graph_from_file(const char *filename, vector<int> &U, vector<int> &V, vector<int> &W) {
    // Open the file for reading
    ifstream infile(filename);
    // A string to store each line of the file
    string line;

    // Three integers to store the source vertex, destination vertex, and weight of each edge
    int u, v, w; 
    // A character to store the first character of each line (to check if it is an edge line)
    char c;
    
    // Read each line of the file
    while (getline(infile, line)) {
        // Create a stringstream from the line
        stringstream ss(line);
        // Read the first character of the line
        ss >> c; 
        
        // If the first character is not 'a', skip the line (it is not an edge line)
        if (c != 'a')
            continue;

        // Read the source vertex, destination vertex, and weight of the edge
        ss >> u >> v >> w;
        // Subtract 1 from the source and destination vertices (to convert from 1-based indexing to 0-based indexing)
        U.push_back(u - 1); 
        V.push_back(v - 1);
        W.push_back(w);
    }

    // Close the file
    infile.close();
}

