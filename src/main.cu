#include "../include/kmeans_cpu.h"
// #include "../include/kmeans_gpu.h"  // Commented out GPU header
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <chrono>
#include <iostream>

// Function to print usage information
void printUsage(const char* programName) {
    printf("Usage: %s [options]\n\n", programName);
    printf("Options:\n");
    printf("  --points=N          Number of points (default: 1000)\n");
    printf("  --centroids=N       Number of centroids (default: 2)\n");
    printf("  --dim=N             Number of dimensions (default: 2)\n");
    printf("  --iterations=N      Maximum iterations (default: 10)\n");
    printf("  --mode=MODE         Operation mode (default: simple)\n");
    printf("                      MODE can be: simple, test, verify\n");
    printf("  --help              Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s --points=5000 --centroids=3 --mode=test\n", programName);
    printf("  %s --mode=verify --iterations=20\n", programName);
}

// Function to parse command line arguments
bool parseArgs(int argc, char** argv, int& num_points, int& num_centroids, 
              int& dim, int& max_iterations, char*& mode) {
    // Default values
    num_points = 1000;
    num_centroids = 2;
    dim = 2;
    max_iterations = 10;
    mode = strdup("simple");
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--points=", 9) == 0) {
            num_points = atoi(argv[i] + 9);
        } else if (strncmp(argv[i], "--centroids=", 12) == 0) {
            num_centroids = atoi(argv[i] + 12);
        } else if (strncmp(argv[i], "--dim=", 6) == 0) {
            dim = atoi(argv[i] + 6);
        } else if (strncmp(argv[i], "--iterations=", 13) == 0) {
            max_iterations = atoi(argv[i] + 13);
        } else if (strncmp(argv[i], "--mode=", 7) == 0) {
            free(mode);
            mode = strdup(argv[i] + 7);
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return false;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return false;
        }
    }
    
    return true;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int num_points, num_centroids, dim, max_iterations;
    char* mode;
    
    if (!parseArgs(argc, argv, num_points, num_centroids, dim, max_iterations, mode)) {
        return 1;
    }
    
    printf("Running k-means clustering with:\n");
    printf("Number of points: %d\n", num_points);
    printf("Number of centroids: %d\n", num_centroids);
    printf("Dimensions: %d\n", dim);
    printf("Max iterations: %d\n", max_iterations);
    printf("Mode: %s\n\n", mode);
    
    // Allocate memory
    float* points = new float[num_points * dim];
    float* cpu_centroids = new float[num_centroids * dim];
    // float* gpu_centroids = new float[num_centroids * dim];  // Commented out GPU memory
    int* cpu_clusters = new int[num_points];
    // int* gpu_clusters = new int[num_points];  // Commented out GPU memory
    
    // Generate random points
    srand(time(NULL));
    for (int i = 0; i < num_points * dim; i++) {
        points[i] = (float)rand() / RAND_MAX * 100.0f;
    }
    
    // Initialize centroids
    for (int i = 0; i < num_centroids * dim; i++) {
        cpu_centroids[i] = points[i];
    }
    
    // Run CPU implementation with timing
    auto start = std::chrono::high_resolution_clock::now();
    kmeans_cpu(points, cpu_centroids, cpu_clusters,
               num_points, num_centroids, dim, max_iterations, 1e-6f);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Print results based on mode
    if (strcmp(mode, "simple") == 0) {
        // Simple mode - just print centroids
        printf("\nCentroids:\n");
        for (int c = 0; c < num_centroids; c++) {
            printf("Centroid %d: ", c);
            for (int d = 0; d < dim; d++) {
                printf("%.6f ", cpu_centroids[c * dim + d]);
            }
            printf("\n");
        }
    } else if (strcmp(mode, "test") == 0) {
        // Test mode - print detailed information
        printf("CPU Time: %ld ms\n", duration.count());
        
        printf("\nCentroids:\n");
        for (int c = 0; c < num_centroids; c++) {
            printf("Centroid %d: ", c);
            for (int d = 0; d < dim; d++) {
                printf("%.6f ", cpu_centroids[c * dim + d]);
            }
            printf("\n");
        }
        
        printf("\nCluster assignments (first 10 points):\n");
        for (int p = 0; p < std::min(10, num_points); p++) {
            printf("Point %d: Cluster %d\n", p, cpu_clusters[p]);
        }
    } else if (strcmp(mode, "verify") == 0) {
        // Verify mode - run verification
        printf("CPU Time: %ld ms\n", duration.count());
        verifyClustering(points, cpu_centroids, cpu_clusters, 
                        num_points, num_centroids, dim);
    }
    
    delete[] points;
    delete[] cpu_centroids;
    // delete[] gpu_centroids;  // Commented out GPU memory cleanup
    delete[] cpu_clusters;
    // delete[] gpu_clusters;  // Commented out GPU memory cleanup
    free(mode);
    
    return 0;
} 