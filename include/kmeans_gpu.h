#ifndef KMEANS_GPU_H
#define KMEANS_GPU_H

#include <cuda_runtime.h>

bool kmeans_cuda(float* points, float* centroids, int* clusters,
                 int num_points, int num_centroids, int dim, int max_iterations, float tolerance,
                 int* iterations);

__device__ float atomicMaxFloat(float* address, float val);
__global__ void initializeCentroidsGPU(float* points, float* centroids, int num_points, int num_centroids, int dim);
__global__ void assignPointsGPU(float* points, float* centroids, int* clusters, int num_points, int num_centroids, int dim);
__global__ void updateCentroidsGPU(float* points, float* centroids, float* newCentroids, int* clusters, int* counts, int* sums, int num_points, int num_centroids, int dim);
__global__ void updatePointsForDimGPU(float* points, int* clusters, int* counts, float* sums, int num_points, int num_dims, int curDim);
__global__ void checkConvergenceGPU(float* centroids, float* newCentroids, int num_centroids, int dim, float tolerance, float* max_diff);
#endif // KMEANS_GPU_H 