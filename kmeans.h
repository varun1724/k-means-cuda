#ifndef KMEANS_H
#define KMEANS_H

// Function declaration for CUDA k-means implementation
void kmeans_cuda(float* h_points, float* h_centroids, int* h_assignments,
                 int num_points, int num_centroids, int dim, int max_iterations);

#endif // KMEANS_H 