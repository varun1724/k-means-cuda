#include "../include/kmeans_cpu.h"
#include "../include/kmeans_gpu.h"
#include "../include/kmeans_utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <chrono>
#include <vector>

#define TOLERANCE 1e-3f

int main(int argc, char** argv) {
    int num_points, num_centroids, dim, max_iterations;
    char* mode;
    
    if (!parseArgs(argc, argv, num_points, num_centroids, dim, max_iterations, mode)) {
        free(mode);
        return 1;
    }
    
    // Allocate memory
    float* points = new float[num_points * dim];
    float* centroids = new float[num_centroids * dim];
    float* gpu_centroids = new float[num_centroids * dim];
    int* clusters = new int[num_points];
    int* gpu_clusters = new int[num_points];
    
    // Initialize points with random values, scaled to 0-100 range
    srand(time(NULL));
    for (int i = 0; i < num_points * dim; i++) {
        points[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    
    int cpu_iterations = 0;
    int gpu_iterations = 0;
    bool cpu_converged = false;
    bool gpu_converged = false;
    
    // Warm up GPU
    if (strcmp(mode, "gpu") != 0) {
        printf("Warm up GPU...\n");
        warmupGPU(num_points, num_centroids, dim, TOLERANCE);
    }
    
    // Run based on mode
    if (strcmp(mode, "cpu") == 0) {
        // Simple CPU-only run
        printf("\nRunning CPU implementation...\n");
        auto start = std::chrono::high_resolution_clock::now();
        cpu_converged = kmeans_cpu(points, centroids, clusters, 
                                 num_points, num_centroids, dim, 
                                 max_iterations, TOLERANCE, &cpu_iterations);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("\nCPU Results:\n");
        printf("Converged: %s in %d iterations\n", cpu_converged ? "Yes" : "No", cpu_iterations);
        printf("Time: %ld ms\n", duration.count());

        printf("\nCPU Centroids:\n");
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < dim; j++) {
                printf("%.6f ", centroids[i * dim + j]);
            }
            printf("\n");
        }

        printClusterPercentages(clusters, num_points, num_centroids, "CPU");
        if (dim == 2) {
            visualizeResults2D(points, clusters, centroids, num_points, num_centroids, dim, "CPU");
        }
    }
    else if (strcmp(mode, "gpu") == 0) {
        // Simple GPU-only run
        printf("\nRunning GPU implementation...\n");
        auto start = std::chrono::high_resolution_clock::now();
        gpu_converged = kmeans_cuda(points, gpu_centroids, gpu_clusters, 
                                  num_points, num_centroids, dim, 
                                  max_iterations, TOLERANCE, &gpu_iterations);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("\nGPU Results:\n");
        printf("Converged: %s in %d iterations\n", gpu_converged ? "Yes" : "No", gpu_iterations);
        printf("Time: %ld ms\n", duration.count());

        printf("\nGPU Centroids:\n");
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < dim; j++) {
                printf("%.6f ", gpu_centroids[i * dim + j]);
            }
            printf("\n");
        }
        
        printClusterPercentages(gpu_clusters, num_points, num_centroids, "GPU");
        if (dim == 2) {
            visualizeResults2D(points, gpu_clusters, gpu_centroids, num_points, num_centroids, dim, "GPU");
        }
    }
    else if (strcmp(mode, "compare") == 0) {
        // Run both CPU and GPU implementations and compare results
        
        // CPU Implementation
        printf("\nRunning CPU implementation...\n");
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_converged = kmeans_cpu(points, centroids, clusters, 
                                 num_points, num_centroids, dim, 
                                 max_iterations, TOLERANCE, &cpu_iterations);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
        
        // GPU Implementation
        printf("\nRunning GPU implementation...\n");
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_converged = kmeans_cuda(points, gpu_centroids, gpu_clusters, 
                                  num_points, num_centroids, dim, 
                                  max_iterations, TOLERANCE, &gpu_iterations);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
        
        // Print results
        printf("\nResults Comparison:\n");
        printf("CPU: Converged %s in %d iterations, Time: %ld ms\n",
               cpu_converged ? "Yes" : "No", cpu_iterations, cpu_duration.count());
        printf("GPU: Converged %s in %d iterations, Time: %ld ms\n",
               gpu_converged ? "Yes" : "No", gpu_iterations, gpu_duration.count());
        
        printf("\nCPU Centroids:\n");
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < dim; j++) { 
                printf("%.6f ", centroids[i * dim + j]);
            }
            printf("\n");
        }

        printf("\nGPU Centroids:\n");
        for (int i = 0; i < num_centroids; i++) {
            for (int j = 0; j < dim; j++) {
                printf("%.6f ", gpu_centroids[i * dim + j]);
            }
            printf("\n");
        }
        
        // Print cluster distributions
        printClusterPercentages(clusters, num_points, num_centroids, "CPU");
        printClusterPercentages(gpu_clusters, num_points, num_centroids, "GPU");
        
        // Visualize if 2D
        if (dim == 2) {
            visualizeResults2D(points, clusters, centroids, num_points, num_centroids, dim, "CPU");
            visualizeResults2D(points, gpu_clusters, gpu_centroids, num_points, num_centroids, dim, "GPU");
        }
    }
    
    delete[] points;
    delete[] centroids;
    delete[] gpu_centroids;
    delete[] clusters;
    delete[] gpu_clusters;
    free(mode);
    
    return 0;
} 