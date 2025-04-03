#include "kmeans_cpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <limits>
#include <cmath>

// Main k-means function
void kmeans_cpu(float* points, float* centroids, int* clusters,
                int num_points, int num_centroids, int dim, int max_iterations, float tolerance) {
    float* newCentroids = new float[num_centroids * dim];
    
    initializeCentroidsCPU(points, centroids, num_points, num_centroids, dim);
    
    int converged_iteration = -1;
    
    for (int i = 0; i < max_iterations; ++i) {
        assignPointsCPU(points, centroids, clusters, num_points, num_centroids, dim);
        updateCentroidsCPU(points, centroids, newCentroids, clusters, num_points, num_centroids, dim);

        // Check for convergence
        float diff = 0.0f;
        for (int c = 0; c < num_centroids; ++c) {
            for (int d = 0; d < dim; ++d) {
                diff += fabs(centroids[c * dim + d] - newCentroids[c * dim + d]);
                if (diff > tolerance) {
                    break;
                }
            }
        }
        if (diff < tolerance) {
            converged_iteration = i + 1;
            break;
        }

        for (int i = 0; i < num_centroids; ++i) {
            for (int d = 0; d < dim; ++d) {
                centroids[i * dim + d] = newCentroids[i * dim + d];
            }
        }
    }

    // Print convergence information
    if (converged_iteration >= 0) {
        printf("Converged at iteration %d (within tolerance %.6f)\n", converged_iteration, tolerance);
    } else {
        printf("Did not converge within %d iterations (tolerance: %.6f)\n", max_iterations, tolerance);
    }

    delete[] newCentroids;
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

// Function to verify the clustering results, taken from the internet
void verifyClustering(float* points, float* centroids, int* clusters, 
                     int num_points, int num_centroids, int dim) {
    // Calculate the total within-cluster sum of squares (WCSS)
    float wcss = 0.0f;
    for (int p = 0; p < num_points; ++p) {
        int cluster = clusters[p];
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = points[p * dim + d] - centroids[cluster * dim + d];
            dist += diff * diff;
        }
        wcss += dist;
    }
    
    // Count points in each cluster
    std::vector<int> cluster_sizes(num_centroids, 0);
    for (int p = 0; p < num_points; ++p) {
        cluster_sizes[clusters[p]]++;
    }
    
    // Print verification results
    printf("\nVerification Results:\n");
    printf("Total within-cluster sum of squares: %.6f\n", wcss);
    printf("Average distance to centroid: %.6f\n", wcss / num_points);
    
    printf("\nCluster sizes:\n");
    for (int c = 0; c < num_centroids; ++c) {
        printf("Cluster %d: %d points (%.2f%%)\n", 
               c, cluster_sizes[c], 
               (float)cluster_sizes[c] / num_points * 100.0f);
    }
    
    // Check for empty clusters
    bool has_empty_clusters = false;
    for (int c = 0; c < num_centroids; ++c) {
        if (cluster_sizes[c] == 0) {
            printf("WARNING: Cluster %d is empty!\n", c);
            has_empty_clusters = true;
        }
    }
    
    if (!has_empty_clusters) {
        printf("All clusters have at least one point assigned.\n");
    }
    
    // Check for convergence stability
    printf("\nChecking convergence stability...\n");
    float* test_centroids = new float[num_centroids * dim];
    int* test_clusters = new int[num_points];
    
    // Copy initial centroids
    for (int i = 0; i < num_centroids * dim; ++i) {
        test_centroids[i] = centroids[i];
    }
    
    // Run one more iteration
    assignPointsCPU(points, test_centroids, test_clusters, num_points, num_centroids, dim);
    
    // Check if assignments changed
    int changed_assignments = 0;
    for (int p = 0; p < num_points; ++p) {
        if (test_clusters[p] != clusters[p]) {
            changed_assignments++;
        }
    }
    
    if (changed_assignments == 0) {
        printf("Convergence is stable: No points changed cluster assignment in an additional iteration.\n");
    } else {
        printf("WARNING: Convergence may not be stable: %d points changed cluster assignment in an additional iteration.\n", 
               changed_assignments);
    }
    
    delete[] test_centroids;
    delete[] test_clusters;
} 
