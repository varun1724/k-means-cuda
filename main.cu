#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"

// Function to generate random points
void generateRandomPoints(float* points, int num_points, int dim) {
    srand(time(NULL));
    for (int i = 0; i < num_points * dim; i++) {
        points[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Function to initialize centroids
void initializeCentroids(float* points, float* centroids, int num_points, int num_centroids, int dim) {
    srand(time(NULL));
    for (int i = 0; i < num_centroids; i++) {
        int idx = rand() % num_points;
        for (int d = 0; d < dim; d++) {
            centroids[i * dim + d] = points[idx * dim + d];
        }
    }
}

int main() {
    // Parameters
    const int num_points = 10000;
    const int num_centroids = 5;
    const int dim = 2;
    const int max_iterations = 100;

    // Allocate memory
    float* points = (float*)malloc(num_points * dim * sizeof(float));
    float* centroids = (float*)malloc(num_centroids * dim * sizeof(float));
    int* assignments = (int*)malloc(num_points * sizeof(int));

    // Generate random points and initialize centroids
    generateRandomPoints(points, num_points, dim);
    initializeCentroids(points, centroids, num_points, num_centroids, dim);

    // Perform k-means clustering
    kmeans_cuda(points, centroids, assignments, num_points, num_centroids, dim, max_iterations);

    // Print results
    printf("Final centroids:\n");
    for (int i = 0; i < num_centroids; i++) {
        printf("Centroid %d: (", i);
        for (int d = 0; d < dim; d++) {
            printf("%.2f", centroids[i * dim + d]);
            if (d < dim - 1) printf(", ");
        }
        printf(")\n");
    }

    // Count points in each cluster
    int* cluster_sizes = (int*)calloc(num_centroids, sizeof(int));
    for (int i = 0; i < num_points; i++) {
        cluster_sizes[assignments[i]]++;
    }

    printf("\nCluster sizes:\n");
    for (int i = 0; i < num_centroids; i++) {
        printf("Cluster %d: %d points\n", i, cluster_sizes[i]);
    }

    // Free memory
    free(points);
    free(centroids);
    free(assignments);
    free(cluster_sizes);

    return 0;
} 