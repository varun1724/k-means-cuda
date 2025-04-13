#include "../include/kmeans_gpu.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(x) do { \
    cudaError_t e = x; \
    if (e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Initialize centroids on GPU
__global__ void initializeCentroidsGPU(float* points, float* centroids, int num_points, int num_centroids, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_centroids) {
        int point_idx = (idx * num_points) / num_centroids;
        for (int d = 0; d < dim; d++) {
            centroids[idx * dim + d] = points[point_idx * dim + d];
        }
    }
}

// Assign points to clusters on GPU
__global__ void assignPointsGPU(float* points, float* centroids, int* clusters, 
                               int num_points, int num_centroids, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        float min_dist = INFINITY;
        int closest_centroid = 0;
        
        // Find the nearest centroid for the current point
        for (int c = 0; c < num_centroids; c++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = points[idx * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = c;
            }
        }
        
        clusters[idx] = closest_centroid;
    }
}

// Assign points to clusters for a specific dimension on GPU
__global__ void updatePointsForDimGPU(float* points, int* clusters, int* counts, float* sums, int num_points, int num_dims, int curDim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points) {
        int cluster = clusters[idx];
        atomicAdd(&sums[cluster * num_dims + curDim], points[idx * num_dims + curDim]);
        if (curDim == 0) {
            atomicAdd(&counts[cluster], 1);
        }
    }
}

// Update centroids on GPU - accumulate sums and counts
__global__ void updateCentroidsGPU(float* points, float* centroids, float* newCentroids, 
                                  int* clusters, int* counts, float* sums, int num_points, int num_centroids, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_centroids) {
        for (int d = 0; d < dim; d++) {
            if (counts[idx] > 0) {
                newCentroids[idx * dim + d] = sums[idx * dim + d] / counts[idx];
            } else {
                newCentroids[idx * dim + d] = centroids[idx * dim + d];
            }
        }
    }
}

// Main k-means function
bool kmeans_cuda(float* points, float* centroids, int* clusters,
                int num_points, int num_centroids, int dim, int max_iterations, float tolerance,
                int* iterations) {
    float *d_points, *d_centroids, *d_newCentroids, *d_sums;
    int *d_clusters, *d_counts;
    bool converged = false;
    *iterations = 0;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_points, num_points * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, num_centroids * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_newCentroids, (num_centroids * dim) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clusters, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, num_centroids * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums, num_centroids * dim * sizeof(float)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_points, points, num_points * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, centroids, num_centroids * dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int block_size = 256;
    int num_blocks_points = (num_points + block_size - 1) / block_size;
    int num_blocks_centroids = (num_centroids + block_size - 1) / block_size;
    
    // Initialize centroids
    initializeCentroidsGPU<<<num_blocks_centroids, block_size>>>(d_points, d_centroids, num_points, num_centroids, dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    do {
        (*iterations)++;
        
        // Reset arrays
        CUDA_CHECK(cudaMemset(d_newCentroids, 0, (num_centroids * dim) * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, num_centroids * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_sums, 0, num_centroids * dim * sizeof(float)));
        
        // Assign points to nearest centroids
        assignPointsGPU<<<num_blocks_points, block_size>>>(d_points, d_centroids, d_clusters, num_points, num_centroids, dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update points to clusters for each dimension
        for (int d = 0; d < dim; d++) {
            updatePointsForDimGPU<<<num_blocks_points, block_size>>>(d_points, d_clusters, d_counts, d_sums, num_points, dim, d);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Update centroids
        updateCentroidsGPU<<<num_blocks_points, block_size>>>(d_points, d_centroids, d_newCentroids, d_clusters, d_counts, d_sums, num_points, num_centroids, dim);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy new centroids back to host for convergence check
        float* newCentroids = new float[num_centroids * dim];
        CUDA_CHECK(cudaMemcpy(newCentroids, d_newCentroids, (num_centroids * dim) * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Check convergence
        float max_diff = 0.0f;
        for (int c = 0; c < num_centroids; c++) {
            for (int d = 0; d < dim; d++) {
                int idx = c * dim + d;
                float diff = fabs(centroids[idx] - newCentroids[idx]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        printf("Max diff: %.6f, Iteration: %d, Tolerance: %.3f\n", max_diff, *iterations, tolerance);

        // Update centroids for next iteration
        for (int i = 0; i < num_centroids * dim; i++) {
            centroids[i] = newCentroids[i];
        }
        
        // Copy updated centroids back to device
        CUDA_CHECK(cudaMemcpy(d_centroids, centroids, num_centroids * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        delete[] newCentroids;
        
        if (max_diff < tolerance) {
            converged = true;
            break;
        }
        
    } while (*iterations < max_iterations);
    
    // Copy final results back to host
    CUDA_CHECK(cudaMemcpy(clusters, d_clusters, num_points * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_newCentroids));
    CUDA_CHECK(cudaFree(d_clusters));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_sums));
    return converged;
} 