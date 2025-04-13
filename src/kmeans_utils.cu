#include "../include/kmeans_utils.h"
#include "../include/kmeans_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// Function to print usage information
void printUsage(const char* programName) {
    printf("Usage: %s [options]\n\n", programName);
    printf("Options:\n");
    printf("  --points=N          Number of points (default: 1000)\n");
    printf("  --centroids=N       Number of centroids (default: 2)\n");
    printf("  --dim=N             Number of dimensions (default: 2)\n");
    printf("  --iterations=N      Maximum iterations (default: 10)\n");
    printf("  --mode=MODE         Operation mode (default: cpu)\n");
    printf("                      MODE can be: cpu, gpu, compare\n");
    printf("  --help              Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s --points=5000 --centroids=3 --mode=cpu\n", programName);
    printf("  %s --mode=compare --iterations=20\n", programName);
}

// Function to parse command line arguments
bool parseArgs(int argc, char** argv, int& num_points, int& num_centroids, 
              int& dim, int& max_iterations, char*& mode) {
    // Default values
    num_points = 1000;
    num_centroids = 2;
    dim = 2;
    max_iterations = 10;
    mode = strdup("cpu");
    
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

// Function to print centroids
void printCentroids(float* centroids, int num_centroids, int dim, const char* label) {
    printf("\n%s Centroids:\n", label);
    for (int c = 0; c < num_centroids; c++) {
        printf("Centroid %d: ", c);
        for (int d = 0; d < dim; d++) {
            printf("%.6f ", centroids[c * dim + d]);
        }
        printf("\n");
    }
}

// Function to print percent of points in each cluster
void printClusterPercentages(int* clusters, int num_points, int num_centroids, const char* label) {
    std::vector<int> counts(num_centroids, 0);
    for (int i = 0; i < num_points; i++) {
        counts[clusters[i]]++;
    }
    printf("\n%s Cluster Percentages:\n", label);
    for (int i = 0; i < num_centroids; i++) {
        float percentage = (float)counts[i] / num_points * 100.0f;
        printf("Cluster %d: %d points (%.2f%%)\n", i, counts[i], percentage);
    }
}

// Function to perform GPU warmup
void warmupGPU(int num_points, int num_centroids, int dim, int tolerance) {
    float *points, *centroids;
    int *clusters;
    
    points = new float[num_points * dim];
    centroids = new float[num_centroids * dim];
    clusters = new int[num_points];
    
    for (int i = 0; i < num_points * dim; i++) {
        points[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < num_centroids * dim; i++) {
        centroids[i] = points[i];
    }
    
    for (int i = 0; i < 3; i++) {
        int warmup_iterations;
        kmeans_cuda(points, centroids, clusters, num_points, num_centroids, dim, 2, tolerance, &warmup_iterations);
    }
    
    delete[] points;
    delete[] centroids;
    delete[] clusters;
    
    cudaDeviceSynchronize();
}

// Function to print a simple 2D scatter plot in terminal
void visualizeResults2D(float* points, int* clusters, float* centroids, 
                       int num_points, int num_centroids, int dim,
                       const char* label) {
    if (dim < 2) return; 
    
    const int GRID_SIZE = 30;
    const int PLOT_POINTS = std::min(1000, num_points);  
    
    float min_x = points[0], max_x = points[0];
    float min_y = points[1], max_y = points[1];
    
    for (int i = 0; i < num_points; i++) {
        min_x = std::min(min_x, points[i * dim]);
        max_x = std::max(max_x, points[i * dim]);
        min_y = std::min(min_y, points[i * dim + 1]);
        max_y = std::max(max_y, points[i * dim + 1]);
    }
    
    float x_padding = (max_x - min_x) * 0.1;
    float y_padding = (max_y - min_y) * 0.1;
    min_x -= x_padding;
    max_x += x_padding;
    min_y -= y_padding;
    max_y += y_padding;
    
    printf("\n%s 2D Visualization:\n", label);
    
    std::vector<std::vector<char>> grid(GRID_SIZE, std::vector<char>(GRID_SIZE, ' '));
    
    for (int i = 0; i < PLOT_POINTS; i++) {
        int x = (int)((points[i * dim] - min_x) / (max_x - min_x) * (GRID_SIZE - 1));
        int y = (int)((points[i * dim + 1] - min_y) / (max_y - min_y) * (GRID_SIZE - 1));
        x = std::min(std::max(x, 0), GRID_SIZE - 1);
        y = std::min(std::max(y, 0), GRID_SIZE - 1);
        grid[GRID_SIZE - 1 - y][x] = '0' + (clusters[i] % 10);  
    }
    
    for (int c = 0; c < num_centroids; c++) {
        int x = (int)((centroids[c * dim] - min_x) / (max_x - min_x) * (GRID_SIZE - 1));
        int y = (int)((centroids[c * dim + 1] - min_y) / (max_y - min_y) * (GRID_SIZE - 1));
        x = std::min(std::max(x, 0), GRID_SIZE - 1);
        y = std::min(std::max(y, 0), GRID_SIZE - 1);
        grid[GRID_SIZE - 1 - y][x] = 'C';  
    }
    
    printf("   ");
    for (int x = 0; x < GRID_SIZE; x++) printf("-");
    printf("\n");
    
    for (int y = 0; y < GRID_SIZE; y++) {
        printf("%2d|", y);
        for (int x = 0; x < GRID_SIZE; x++) {
            printf("%c", grid[y][x]);
        }
        printf("|\n");
    }
    
    printf("   ");
    for (int x = 0; x < GRID_SIZE; x++) printf("-");
    printf("\n");
    
    printf("\nLegend:\n");
    printf("C: Centroid\n");
    printf("0-9: Points (number indicates cluster)\n");
} 