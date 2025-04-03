#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include <cuda_runtime.h>

// Function declarations for GPU k-means implementation
void kmeans_cuda(float* points, float* centroids, int* clusters,
                 int num_points, int num_centroids, int dim, int max_iterations, int tolerance=1e-4);

// Helper functions
void initializeCentroidsGPU(float* points, float* centroids, int num_points, int num_centroids, int dim);
void assignPointsGPU(float* points, float* centroids, int* clusters, int num_points, int num_centroids, int dim);
void updateCentroidsGPU(float* points, float* centroids, int* clusters, int num_points, int num_centroids, int dim);

#endif // KMEANS_GPU_H 