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

// Atomic operation for float values
__device__ float atomicMaxFloat(float* addr, float val) {
    unsigned int* address_as_uint = (unsigned int*)addr;
    unsigned int old = *address_as_uint;
    unsigned int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
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
    extern __shared__ float shared_centroids[];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 4
    for (int c = threadIdx.x; c < num_centroids * dim; c += blockDim.x) {
        if (c < num_centroids * dim) {
            shared_centroids[c] = centroids[c];
        }
    }
    __syncthreads();
    
    if (idx < num_points) {
        float min_dist = INFINITY;
        int closest_centroid = 0;
        
        // Register loads
        float point_coords[32];  // Max dim is 32
        #pragma unroll 4
        for (int d = 0; d < dim; d++) {
            point_coords[d] = points[idx * dim + d];
        }
        
        for (int c = 0; c < num_centroids; c++) {
            float dist = 0.0f;
            
            #pragma unroll 4
            for (int d = 0; d < dim; d++) {
                float diff = point_coords[d] - shared_centroids[c * dim + d];
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

// Update points kernel 
__global__ void updatePointsForDimGPU(float* points, int* clusters, int* counts, float* sums, 
                                     int num_points, int num_dims, int curDim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_points) {
        int cluster = clusters[idx];
        float point_val = points[idx * num_dims + curDim];
        
        __shared__ float shared_sums[32];  // Max clusters is 32
        __shared__ int shared_counts[32];
        
        if (threadIdx.x < 32) {
            shared_sums[threadIdx.x] = 0.0f;
            shared_counts[threadIdx.x] = 0;
        }
        __syncthreads();
        
        atomicAdd(&shared_sums[cluster], point_val);
        // Only update count for first dimension -> each point only added once
        if (curDim == 0) {
            atomicAdd(&shared_counts[cluster], 1);
        }
        __syncthreads();
        
        // Only one thread per block updates global memory
        if (threadIdx.x < 32) {
            if (shared_sums[threadIdx.x] != 0.0f) {
                atomicAdd(&sums[threadIdx.x * num_dims + curDim], shared_sums[threadIdx.x]);
            }
            if (curDim == 0 && shared_counts[threadIdx.x] > 0) {
                atomicAdd(&counts[threadIdx.x], shared_counts[threadIdx.x]);
            }
        }
    }
}

// Update centroids on GPU 
__global__ void updateCentroidsGPU(float* points, float* centroids, float* newCentroids, 
                                  int* clusters, int* counts, float* sums, 
                                  int num_points, int num_centroids, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_centroids) {
        int count = counts[idx];
        if (count > 0) {
            double inv_count = 1.0 / count;
            
            #pragma unroll 4
            for (int d = 0; d < dim; d++) {
                newCentroids[idx * dim + d] = (float)(sums[idx * dim + d] * inv_count);
            }
        } else {
            #pragma unroll 4
            for (int d = 0; d < dim; d++) {
                newCentroids[idx * dim + d] = centroids[idx * dim + d];
            }
        }
    }
}

// Check convergence 
__global__ void checkConvergenceGPU(float* centroids, float* newCentroids, 
                                   int num_centroids, int dim, float tolerance, float* max_diff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_centroids) {
        float local_max_diff = 0.0f;
        
        #pragma unroll 4
        for (int d = 0; d < dim; d++) {
            float old_val = centroids[idx * dim + d];
            float new_val = newCentroids[idx * dim + d];
            
            float abs_val = fmaxf(fabsf(old_val), fabsf(new_val));
            float diff;
            if (abs_val > 1e-6f) {
                diff = fabsf(new_val - old_val) / abs_val;
            } else {
                diff = fabsf(new_val - old_val);
            }
            local_max_diff = fmaxf(local_max_diff, diff);
        }
        
        if (local_max_diff > 0.0f) {
            atomicMaxFloat(max_diff, local_max_diff);
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

    int block_size = 256;
    int num_blocks_points = (num_points + block_size - 1) / block_size;
    int num_blocks_centroids = (num_centroids + block_size - 1) / block_size;

    *iterations = 0;
    bool converged = false;    

    while (!converged && *iterations < max_iterations) {
        (*iterations)++;

        CUDA_CHECK(cudaMemsetAsync(d_newCentroids, 0, num_centroids * dim * sizeof(float), streams[0]));
        CUDA_CHECK(cudaMemsetAsync(d_counts, 0, num_centroids * sizeof(int), streams[0]));
        CUDA_CHECK(cudaMemsetAsync(d_sums, 0, num_centroids * dim * sizeof(float), streams[0]));
        
        size_t shared_mem_size = num_centroids * dim * sizeof(float);
        assignPointsGPU<<<num_blocks_points, block_size, shared_mem_size, streams[0]>>>(
            d_points, d_centroids, d_clusters, num_points, num_centroids, dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));

        for (int d = 0; d < dim; d++) {
            updatePointsForDimGPU<<<num_blocks_points, block_size, 0, streams[1]>>>(
                d_points, d_clusters, d_counts, d_sums, num_points, dim, d);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
            
        updateCentroidsGPU<<<num_blocks_points, block_size, 0, streams[1]>>>(
            d_points, d_centroids, d_newCentroids, d_clusters, d_counts, d_sums, 
            num_points, num_centroids, dim);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));

        CUDA_CHECK(cudaMemsetAsync(d_max_diff, 0, sizeof(float), streams[2]));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
        
        checkConvergenceGPU<<<num_blocks_centroids, block_size, 0, streams[2]>>>(
            d_centroids, d_newCentroids, num_centroids, dim, tolerance, d_max_diff);
        CUDA_CHECK(cudaGetLastError());

        float h_max_diff;
        CUDA_CHECK(cudaMemcpyAsync(&h_max_diff, d_max_diff, sizeof(float), 
                                  cudaMemcpyDeviceToHost, streams[2]));
        CUDA_CHECK(cudaStreamSynchronize(streams[2]));
        
        converged = (h_max_diff <= tolerance);
        
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