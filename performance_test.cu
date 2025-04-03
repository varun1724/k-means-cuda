#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
// #include "kmeans_gpu.h"  // Commented out GPU header
#include "kmeans_cpu.h"

// Function to generate test data with better numerical stability
void generateTestData(float* points, float* centroids, int num_points, int num_centroids, int dim) {
    srand(time(NULL));
    
    // Generate well-separated centroids
    for (int c = 0; c < num_centroids; c++) {
        for (int d = 0; d < dim; d++) {
            // Space centroids evenly in the [0, 100] range
            centroids[c * dim + d] = (float)((100.0 * c) / (num_centroids - 1));
        }
    }
    
    // Generate points around centroids with controlled noise
    for (int p = 0; p < num_points; p++) {
        int c = rand() % num_centroids;
        for (int d = 0; d < dim; d++) {
            // Add small random noise (-1 to 1)
            float noise = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            points[p * dim + d] = centroids[c * dim + d] + noise * 2.0f;
        }
    }
}

// Function to verify results with detailed reporting
bool verifyResults(float* cpu_centroids, float* gpu_centroids, int* cpu_clusters, int* gpu_clusters,
                  int num_centroids, int num_points, int dim, float tolerance = 1e-5) {
    bool match = true;
    bool first_mismatch = true;
    
    // Check centroids
    for (int c = 0; c < num_centroids; c++) {
        for (int d = 0; d < dim; d++) {
            float diff = fabs(cpu_centroids[c * dim + d] - gpu_centroids[c * dim + d]);
            if (diff > tolerance) {
                if (first_mismatch) {
                    printf("\nFirst mismatch found:\n");
                    printf("Centroid [%d][%d]:\n", c, d);
                    printf("  CPU: %.6f\n", cpu_centroids[c * dim + d]);
                    printf("  GPU: %.6f\n", gpu_centroids[c * dim + d]);
                    printf("  Diff: %.6f\n", diff);
                    first_mismatch = false;
                }
                match = false;
            }
        }
    }
    
    // Check clusters
    std::vector<int> cpu_cluster_sizes(num_centroids, 0);
    std::vector<int> gpu_cluster_sizes(num_centroids, 0);
    
    for (int p = 0; p < num_points; p++) {
        cpu_cluster_sizes[cpu_clusters[p]]++;
        gpu_cluster_sizes[gpu_clusters[p]]++;
        
        if (cpu_clusters[p] != gpu_clusters[p]) {
            if (first_mismatch) {
                printf("\nFirst mismatch found:\n");
                printf("Point %d:\n", p);
                printf("  CPU cluster: %d\n", cpu_clusters[p]);
                printf("  GPU cluster: %d\n", gpu_clusters[p]);
                first_mismatch = false;
            }
            match = false;
        }
    }
    
    // Check cluster sizes
    for (int c = 0; c < num_centroids; c++) {
        if (cpu_cluster_sizes[c] != gpu_cluster_sizes[c]) {
            if (first_mismatch) {
                printf("\nFirst mismatch found:\n");
                printf("Cluster %d size:\n", c);
                printf("  CPU size: %d\n", cpu_cluster_sizes[c]);
                printf("  GPU size: %d\n", gpu_cluster_sizes[c]);
                first_mismatch = false;
            }
            match = false;
        }
    }
    
    return match;
}

int main() {
    // Test parameters - reduced to focus on core issues
    const std::vector<int> num_points_list = {1000};  // Start with smaller dataset
    const std::vector<int> num_centroids_list = {2};  // Start with 2 centroids
    const std::vector<int> dim_list = {2};           // Start with 2 dimensions
    const int max_iterations = 10;                   // Reduce iterations for testing
    
    printf("Running CPU-only performance test...\n");
    printf("================================\n\n");
    
    for (int num_points : num_points_list) {
        for (int num_centroids : num_centroids_list) {
            for (int dim : dim_list) {
                printf("Configuration: %d points, %d centroids, %d dimensions\n", 
                       num_points, num_centroids, dim);
                
                // Allocate memory
                float *points = new float[num_points * dim];
                float *cpu_centroids = new float[num_centroids * dim];
                // float *gpu_centroids = new float[num_centroids * dim];  // Commented out GPU memory
                int *cpu_clusters = new int[num_points];
                // int *gpu_clusters = new int[num_points];  // Commented out GPU memory
                
                // Generate test data
                generateTestData(points, cpu_centroids, num_points, num_centroids, dim);
                // memcpy(gpu_centroids, cpu_centroids, num_centroids * dim * sizeof(float));  // Commented out GPU copy
                
                // CPU implementation timing
                auto cpu_start = std::chrono::high_resolution_clock::now();
                kmeans_cpu(points, cpu_centroids, cpu_clusters,
                          num_points, num_centroids, dim, max_iterations);
                auto cpu_end = std::chrono::high_resolution_clock::now();
                auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
                
                // GPU implementation timing - commented out
                /*
                auto gpu_start = std::chrono::high_resolution_clock::now();
                kmeans_cuda(points, gpu_centroids, gpu_clusters,
                           num_points, num_centroids, dim, max_iterations);
                auto gpu_end = std::chrono::high_resolution_clock::now();
                auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);
                */
                
                // Print CPU results
                printf("CPU Time: %ld ms\n", cpu_duration.count());
                // printf("GPU Time: %ld ms\n", gpu_duration.count());  // Commented out GPU timing
                // printf("Speedup: %.2fx\n", (float)cpu_duration.count() / gpu_duration.count());  // Commented out speedup
                // printf("Results Match: %s\n", results_match ? "Yes" : "No");  // Commented out comparison
                
                // Print CPU centroids
                printf("\nCPU centroids:\n");
                for (int c = 0; c < num_centroids; c++) {
                    printf("Centroid %d: ", c);
                    for (int d = 0; d < dim; d++) {
                        printf("%.6f ", cpu_centroids[c * dim + d]);
                    }
                    printf("\n");
                }
                
                // Print cluster assignments
                printf("\nCluster assignments (first 10 points):\n");
                for (int p = 0; p < std::min(10, num_points); p++) {
                    printf("Point %d: Cluster %d\n", p, cpu_clusters[p]);
                }
                
                // Verify the clustering results
                verifyClustering(points, cpu_centroids, cpu_clusters, 
                               num_points, num_centroids, dim);
                
                printf("\n----------------------------------------\n\n");
                
                // Free memory
                delete[] points;
                delete[] cpu_centroids;
                // delete[] gpu_centroids;  // Commented out GPU memory cleanup
                delete[] cpu_clusters;
                // delete[] gpu_clusters;  // Commented out GPU memory cleanup
            }
        }
    }
    
    return 0;
} 