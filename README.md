# CUDA-Accelerated K-Means Clustering

This project implements a CUDA-accelerated version of the k-means clustering algorithm. The implementation uses parallel processing on the GPU to significantly speed up the clustering process.

## Features

- Parallel distance computation between points and centroids
- Efficient point assignment to nearest centroids
- Parallel centroid updates
- Optimized memory access patterns
- Support for multi-dimensional data

## Requirements

- CUDA Toolkit (version 10.0 or higher)
- NVIDIA GPU with compute capability 6.0 or higher
- GNU Make

## Building

To build the project, simply run:

```bash
make
```

This will create an executable named `kmeans`.

## Usage

The program will automatically generate random 2D points and perform k-means clustering with the following default parameters:
- Number of points: 10,000
- Number of clusters: 5
- Dimensions: 2
- Maximum iterations: 100

To run the program:

```bash
./kmeans
```

## Implementation Details

The implementation consists of three main CUDA kernels:

1. `computeDistances`: Computes the squared Euclidean distance between each point and each centroid
2. `assignPoints`: Assigns each point to its nearest centroid
3. `updateCentroids`: Updates the centroid positions based on the current assignments

## Performance

The CUDA implementation provides significant speedup compared to a CPU implementation, especially for large datasets. The exact speedup depends on:
- Number of points
- Number of clusters
- Data dimensionality
- GPU architecture

## License

This project is open source and available under the MIT License.
