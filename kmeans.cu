#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "kmeans.h"

// CUDA kernel for computing distances between points and centroids
__global__ void computeDistances(float* points, float* centroids, float* distances, 
                                int num_points, int num_centroids, int dim) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int centroid_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (point_idx < num_points && centroid_idx < num_centroids) {
        float sum = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = points[point_idx * dim + d] - centroids[centroid_idx * dim + d];
            sum += diff * diff;
        }
        distances[point_idx * num_centroids + centroid_idx] = sum;
    }
}

// CUDA kernel for assigning points to nearest centroids
__global__ void assignPoints(float* distances, int* assignments, int num_points, int num_centroids) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (point_idx < num_points) {
        float min_dist = distances[point_idx * num_centroids];
        int min_idx = 0;
        
        for (int c = 1; c < num_centroids; c++) {
            float dist = distances[point_idx * num_centroids + c];
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = c;
            }
        }
        assignments[point_idx] = min_idx;
    }
}

// CUDA kernel for updating centroids
__global__ void updateCentroids(float* points, int* assignments, float* centroids, 
                               int* counts, int num_points, int num_centroids, int dim) {
    int centroid_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (centroid_idx < num_centroids && dim_idx < dim) {
        float sum = 0.0f;
        int count = 0;
        
        for (int p = 0; p < num_points; p++) {
            if (assignments[p] == centroid_idx) {
                sum += points[p * dim + dim_idx];
                count++;
            }
        }
        
        if (count > 0) {
            centroids[centroid_idx * dim + dim_idx] = sum / count;
        }
        counts[centroid_idx] = count;
    }
}

// Host function to perform k-means clustering
void kmeans_cuda(float* h_points, float* h_centroids, int* h_assignments,
                 int num_points, int num_centroids, int dim, int max_iterations) {
    
    // Allocate device memory
    float *d_points, *d_centroids, *d_distances;
    int *d_assignments, *d_counts;
    
    cudaMalloc(&d_points, num_points * dim * sizeof(float));
    cudaMalloc(&d_centroids, num_centroids * dim * sizeof(float));
    cudaMalloc(&d_distances, num_points * num_centroids * sizeof(float));
    cudaMalloc(&d_assignments, num_points * sizeof(int));
    cudaMalloc(&d_counts, num_centroids * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_points, h_points, num_points * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, num_centroids * dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((num_points + blockDim.x - 1) / blockDim.x,
                 (num_centroids + blockDim.y - 1) / blockDim.y);
    
    // Main k-means loop
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute distances
        computeDistances<<<gridDim, blockDim>>>(d_points, d_centroids, d_distances,
                                              num_points, num_centroids, dim);
        
        // Assign points to nearest centroids
        dim3 assignBlockDim(256);
        dim3 assignGridDim((num_points + assignBlockDim.x - 1) / assignBlockDim.x);
        assignPoints<<<assignGridDim, assignBlockDim>>>(d_distances, d_assignments,
                                                      num_points, num_centroids);
        
        // Update centroids
        dim3 updateBlockDim(32, 32);
        dim3 updateGridDim((num_centroids + updateBlockDim.x - 1) / updateBlockDim.x,
                          (dim + updateBlockDim.y - 1) / updateBlockDim.y);
        updateCentroids<<<updateGridDim, updateBlockDim>>>(d_points, d_assignments, d_centroids,
                                                         d_counts, num_points, num_centroids, dim);
    }
    
    // Copy results back to host
    cudaMemcpy(h_assignments, d_assignments, num_points * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, num_centroids * dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_distances);
    cudaFree(d_assignments);
    cudaFree(d_counts);
} 