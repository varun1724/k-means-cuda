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

// Atomic max for float
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int expected;
    do {
        expected = old;
        old = atomicCAS(address_as_int, expected,
            __float_as_int(max(__int_as_float(expected), val)));
    } while (expected != old);
    return __int_as_float(old);
}

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

// Check if the centroids have converged
__global__ void checkConvergenceGPU(float* centroids, float* newCentroids, int num_centroids, int dim, float tolerance, float* max_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_centroids) {
        for (int d = 0; d < dim; d++) {
            float diff = fabs(centroids[idx * dim + d] - newCentroids[idx * dim + d]);
            atomicMaxFloat(max_diff, diff);
        }
    }    
}


// Main k-means function
bool kmeans_cuda(float* points, float* centroids, int* clusters, 
                 int num_points, int num_centroids, int dim, 
                 int max_iterations, float tolerance, int* iterations) {
    const int NUM_STREAMS = 3;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    float *d_points, *d_centroids, *d_newCentroids, *d_sums, *d_max_diff;
    int *d_clusters, *d_counts;
    
    CUDA_CHECK(cudaMalloc(&d_points, num_points * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, num_centroids * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_newCentroids, num_centroids * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_clusters, num_points * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts, num_centroids * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums, num_centroids * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max_diff, sizeof(float)));

    CUDA_CHECK(cudaMemcpyAsync(d_points, points, num_points * dim * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(d_centroids, centroids, num_centroids * dim * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[1]));

    // Calculate grid and block dimensions
    int block_size = 256;
    int num_blocks_points = (num_points + block_size - 1) / block_size;
    int num_blocks_centroids = (num_centroids + block_size - 1) / block_size;

    *iterations = 0;
    bool converged = false;    

    while (!converged && *iterations < max_iterations) {
        (*iterations)++;

        // Stream 0: Reset arrays and assign points
        CUDA_CHECK(cudaMemsetAsync(d_newCentroids, 0, num_centroids * dim * sizeof(float), streams[0]));
        CUDA_CHECK(cudaMemsetAsync(d_counts, 0, num_centroids * sizeof(int), streams[0]));
        CUDA_CHECK(cudaMemsetAsync(d_sums, 0, num_centroids * dim * sizeof(float), streams[0]));
        assignPointsGPU<<<num_blocks_points, block_size, 0, streams[0]>>>(
            d_points, d_centroids, d_clusters, num_points, num_centroids, dim);

        // Stream 1: Handle dimension updates (after stream 0 completes)
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        for (int d = 0; d < dim; d++) {
            updatePointsForDimGPU<<<num_blocks_points, block_size, 0, streams[1]>>>(
                d_points, d_clusters, d_counts, d_sums, num_points, dim, d);
        }
        updateCentroidsGPU<<<num_blocks_points, block_size, 0, streams[1]>>>(
            d_points, d_centroids, d_newCentroids, d_clusters, d_counts, d_sums, 
            num_points, num_centroids, dim);

        // Stream 2: Handle convergence check (can start as soon as centroids are updated)
        CUDA_CHECK(cudaMemsetAsync(d_max_diff, 0, sizeof(float), streams[2]));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        checkConvergenceGPU<<<num_blocks_centroids, block_size, 0, streams[2]>>>(
            d_centroids, d_newCentroids, num_centroids, dim, tolerance, d_max_diff);

        float h_max_diff;
        CUDA_CHECK(cudaMemcpyAsync(&h_max_diff, d_max_diff, sizeof(float), 
                                  cudaMemcpyDeviceToHost, streams[2]));
        
        // Only need to synchronize stream 2 for convergence check
        CUDA_CHECK(cudaStreamSynchronize(streams[2]));

        // printf("Max diff: %.6f, Iteration: %d, Tolerance: %.3f\n", h_max_diff, *iterations, tolerance);
        
        converged = (h_max_diff <= tolerance);
        
        // Stream 0: Update centroids for next iteration (can happen while convergence is being checked)
        if (!converged) {
            CUDA_CHECK(cudaMemcpyAsync(d_centroids, d_newCentroids, 
                                     num_centroids * dim * sizeof(float),
                                     cudaMemcpyDeviceToDevice, streams[0]));
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(centroids, d_centroids, num_centroids * dim * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[0]));
    CUDA_CHECK(cudaMemcpyAsync(clusters, d_clusters, num_points * sizeof(int), 
                              cudaMemcpyDeviceToHost, streams[1]));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_newCentroids));
    CUDA_CHECK(cudaFree(d_clusters));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_max_diff));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    return converged;
} 