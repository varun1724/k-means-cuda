#ifndef KMEANS_UTILS_H
#define KMEANS_UTILS_H

#include <vector>

void printUsage(const char* programName);
bool parseArgs(int argc, char** argv, int& num_points, int& num_centroids, 
              int& dim, int& max_iterations, char*& mode);

void printCentroids(float* centroids, int num_centroids, int dim, const char* label);
void printClusterPercentages(int* clusters, int num_points, int num_centroids, const char* label);
void visualizeResults2D(float* points, int* clusters, float* centroids, 
                       int num_points, int num_centroids, int dim,
                       const char* label);

void warmupGPU(int num_points, int num_centroids, int dim, int tolerance);

#endif // KMEANS_UTILS_H 