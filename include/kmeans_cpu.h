#ifndef KMEANS_CPU_H
#define KMEANS_CPU_H

#include <vector>
#include <cuda_runtime.h>

bool kmeans_cpu(float* points, float* centroids, int* clusters,
                int num_points, int num_centroids, int dim, int max_iterations, float tolerance,
                int* iterations);

void initializeCentroidsCPU(float* points, float* centroids, int num_points, int num_centroids, int dim);
void assignPointsCPU(float* points, float* centroids, int* clusters, int num_points, int num_centroids, int dim);
void updateCentroidsCPU(float* points, float* centroids, float* newCentroids, int* clusters, int num_points, int num_centroids, int dim);

#endif // KMEANS_CPU_H 