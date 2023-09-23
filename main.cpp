#include "graph_io.h"
#include "floyd_warshall.h"
#include <vector>
#include <iostream>


using namespace std;

#define INF 999999999

int main() {
    vector<int> u, v, w;
    int edges_count, vertices_count;
    ReadGraphFromFile("graph.txt", u, v, w, edges_count, vertices_count);

    int* distance_matrix;
    InitDistanceMatrix(distance_matrix, vertices_count, edges_count, u.data(), v.data(), w.data());

    FloydWarshall(distance_matrix, vertices_count);

    int value;
    for(int i = 0; i < vertices_count; i++){
        for(int j = 0; j < vertices_count; j++){
            value = distance_matrix[i * vertices_count + j];
            cout << (value == INF ? "INF" : to_string(value)) << " ";
        }
        cout << endl;
    }

    return 0;
}
