#include "../include/kmeans_cpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits>
#include <cmath>

// Main k-means function
bool kmeans_cpu(float* points, float* centroids, int* clusters,
                int num_points, int num_centroids, int dim, int max_iterations, float tolerance,
                int* iterations) {
    float* newCentroids = new float[num_centroids * dim];
    bool converged = false;
    *iterations = 0;
    
    initializeCentroidsCPU(points, centroids, num_points, num_centroids, dim);
    
    for (*iterations = 0; *iterations < max_iterations; (*iterations)++) {
        assignPointsCPU(points, centroids, clusters, num_points, num_centroids, dim);
        updateCentroidsCPU(points, centroids, newCentroids, clusters, num_points, num_centroids, dim);

        // Check for convergence
        float max_diff = 0.0f;
        for (int c = 0; c < num_centroids; c++) {
            for (int d = 0; d < dim; d++) {
                float diff = fabs(centroids[c * dim + d] - newCentroids[c * dim + d]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        // Update centroids for next iteration
        for (int i = 0; i < num_centroids * dim; i++) {
            centroids[i] = newCentroids[i];
        }

        printf("Max diff: %.6f, Iteration: %d, Tolerance: %.3f\n", max_diff, *iterations, tolerance);

        if (max_diff < tolerance) {
            converged = true;
            (*iterations)++;
            break;
        }
    }

    delete[] newCentroids;
    return converged;
}

// Initialize centroids
void initializeCentroidsCPU(float* points, float* centroids, int num_points, int num_centroids, int dim) {
    srand(time(NULL));

    std::vector<int> indices(num_points);
    for (int i = 0; i < num_points; ++i) {
        indices[i] = i;
    }
    for (int i = num_points - 1; i > 0; --i) {
        int j = rand() % (i + 1); 
        std::swap(indices[i], indices[j]);
    }

    for (int i = 0; i < num_centroids; ++i) {
        for (int d = 0; d < dim; ++d) {
            centroids[i * dim + d] = points[indices[i] * dim + d];
        }
    }
}

// Assign points to nearest centroids
void assignPointsCPU(float* points, float* centroids, int* clusters, int num_points, int num_centroids, int dim) {
    for (int i = 0; i < num_points; ++i) {
        float min_dist = std::numeric_limits<float>::infinity();

        for (int c = 0; c < num_centroids; ++c) {
            float dist = 0.0f;
            for (int d = 0; d < dim; ++d) {
                float diff = points[i * dim + d] - centroids[c * dim + d];
                dist += diff * diff;
            }
            
            if (dist < min_dist) {
                min_dist = dist;
                clusters[i] = c;
            }
        }
    }    
}

// Update centroids based on assigned points
void updateCentroidsCPU(float* points, float* centroids, float* newCentroids, int* clusters, int num_points, int num_centroids, int dim) {
    for (int i = 0; i < num_centroids; ++i) {
        for (int d = 0; d < dim; ++d) {
            float sum = 0.0f;
            int count = 0;
            
            for (int p = 0; p < num_points; ++p) {
                if (clusters[p] == i) {
                    sum += points[p * dim + d];
                    ++count;
                }
            }

            if (count > 0) {
                newCentroids[i * dim + d] = sum / count;
            } else {
                // If no points are assigned to this centroid, keep the old centroid
                newCentroids[i * dim + d] = centroids[i * dim + d];
            }
        }
    }
} 
