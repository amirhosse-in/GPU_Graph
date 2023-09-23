#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include "graph_io.h"

#define INF 999999999

using namespace std;

void ReadGraphFromFile(const char *filename, vector<int> &u, vector<int> &v, vector<int> &w, int &edges_count, int &vertices_count) {
    
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


void WriteFloydWarshallResultToFile(const int* distance_matrix, const int vertices_count, const string& filename) {
    // Open the file for writing
    ofstream file(filename);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    // Declare a variable to store the distance value
    int value;

    // Iterate through the rows of the distance matrix
    for(int i = 0; i < vertices_count; i++){
        // Iterate through the columns of the distance matrix
        for(int j = 0; j < vertices_count; j++){
            // Get the distance value for the pair of vertices (i, j)
            value = distance_matrix[i * vertices_count + j];
            
            // Write the distance value to the file, using "INF" for infinity
            file << (value == INF ? "INF" : to_string(value)) << "\t";
        }
        // Start a new line for the next row
        file << endl;
    }
    
    // Close the file
    file.close();
}