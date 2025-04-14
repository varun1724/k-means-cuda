# K-Means Clustering Implementation

This project implements the K-Means clustering algorithm with both CPU and GPU (CUDA) implementations. The project provides a comprehensive comparison between CPU and GPU implementations, demonstrating the challenges and benefits of parallel computing in clustering algorithms.

## Project Structure

```
k-means-cuda/
├── include/                # Header files
│   ├── kmeans_cpu.h       # CPU implementation declarations
│   ├── kmeans_gpu.h       # GPU implementation declarations
│   └── kmeans_utils.h     # Utility function declarations
├── src/                   # Source files
│   ├── main.cu           # Main program
│   ├── kmeans_cpu.cu     # CPU implementation
│   ├── kmeans_gpu.cu     # GPU implementation
│   └── kmeans_utils.cu   # Utility functions implementation
├── Makefile              # Build configuration
└── README.md             # This file
```

## Features

- **Dual Implementation**: Both CPU and GPU implementations of K-Means clustering
- **Multiple Operation Modes**: CPU-only, GPU-only, and comparison modes
- **Configurable Parameters**: Adjustable number of points, centroids, dimensions, and iterations
- **Performance Analysis**: Tools to compare CPU and GPU performance
- **Convergence Reporting**: Detailed convergence status and iteration counts
- **Visualization**: 2D visualization support for better understanding of clustering results

## Implementation Details

### CPU Implementation
- Sequential implementation using standard C++
- Direct floating-point operations for high precision
- Simple and efficient centroid updates
- Straightforward convergence detection

### GPU Implementation
- Parallel implementation using CUDA
- Optimized memory access patterns
- Shared memory utilization for faster computation
- Efficient parallel reduction for centroid updates
- Careful handling of floating-point atomic operations

### Utility Functions
- Centralized helper functions for both implementations
- Robust comparison tools for CPU and GPU results
- Visualization utilities for 2D data
- Command-line argument parsing and validation

## Building the Project

### Prerequisites

- CUDA Toolkit (version 11.0 or higher)
- C++11 compatible compiler
- Make

### Build Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/k-means-cuda.git
   cd k-means-cuda
   ```

2. Build the project:
   ```bash
   make clean
   make
   ```

## Usage

The program provides three modes of operation:

### CPU Mode
Runs only the CPU implementation and displays results:
```bash
./kmeans --mode=cpu [options]
```

### GPU Mode
Runs only the GPU implementation and displays results:
```bash
./kmeans --mode=gpu [options]
```

### Compare Mode
Performs a detailed comparison between CPU and GPU implementations. It shows:
- Timing information for both implementations
- Detailed centroid values for both implementations
- Cluster distribution analysis
- Visualization of results (for 2D data)

Example:
```bash
./kmeans --mode=compare --points=1000000 --centroids=5 --dim=3 --iterations=100
```

### Available Options

- `--points=N`: Number of points (default: 1000)
- `--centroids=N`: Number of centroids (default: 2)
- `--dim=N`: Number of dimensions (default: 2)
- `--iterations=N`: Maximum iterations (default: 10)
- `--mode=MODE`: Operation mode (cpu/gpu/compare)
- `--help`: Show help message

### Example Commands

1. Basic CPU clustering with visualization:
   ```bash
   ./kmeans --mode=cpu --points=100000 --centroids=3 --dim=2 --iterations=1000
   ```

2. GPU clustering with large dataset:
   ```bash
   ./kmeans --mode=gpu --points=1000000 --centroids=5 --dim=3 --iterations=100
   ```

3. Performance comparison:
   ```bash
   ./kmeans --mode=compare --points=500000 --centroids=4 --dim=2 --iterations=100
   ```

## Output Information

For each mode, the program provides:

### CPU/GPU Mode
- Convergence status and number of iterations
- Execution time
- Final centroid positions
- Cluster size distribution
- 2D visualization (if dim=2)

### Compare Mode
- Convergence status for both implementations
- Execution time comparison
- Detailed centroid positions for both implementations
- Cluster size distributions for both implementations
- 2D visualizations for both implementations (if dim=2)

### Sample Output

#### Example 1: 3D Dataset (1 million points)
```bash
$ ./kmeans --mode=compare --points=1000000 --centroids=5 --dim=3 --iterations=100

Results Comparison:
CPU: Converged Yes in 79 iterations, Time: 5665 ms
GPU: Converged Yes in 94 iterations, Time: 224 ms
Speed up: 25.290178

CPU Centroids:
73.518631 73.607964 26.544338 
26.445738 26.559889 73.514366 
23.278177 76.565804 49.854202 
50.270317 23.367228 23.453608 
76.506493 50.128906 76.700531 

GPU Centroids:
73.404716 26.572279 73.645042 
26.369879 73.275757 26.314873 
76.550034 76.596176 49.661751 
23.329519 50.333988 76.672073 
50.079372 23.205437 23.543196 

CPU Cluster Percentages:
Cluster 0: 172326 points (17.23%)
Cluster 1: 172697 points (17.27%)
Cluster 2: 218495 points (21.85%)
Cluster 3: 219463 points (21.95%)
Cluster 4: 217019 points (21.70%)

GPU Cluster Percentages:
Cluster 0: 172199 points (17.22%)
Cluster 1: 172332 points (17.23%)
Cluster 2: 218623 points (21.86%)
Cluster 3: 218282 points (21.83%)
Cluster 4: 218564 points (21.86%)
```

#### Performance Benchmarks

Multiple runs with 1M points, 5 centroids, 3 dimensions:

| Run | CPU Iterations | CPU Time (ms) | CPU ms/iter | GPU Iterations | GPU Time (ms) | GPU ms/iter | Speedup |
|-----|---------------|---------------|-------------|----------------|----------------|-------------|---------|
| 1   | 65           | 5026          | 77.32       | 56            | 33            | 0.59        | 152.30x |
| 2   | 69           | 5058          | 73.30       | 72            | 40            | 0.56        | 126.45x |
| 3   | 87           | 6568          | 75.49       | 66            | 37            | 0.56        | 177.51x |
| 4   | 83           | 6351          | 76.52       | 49            | 30            | 0.61        | 211.70x |

Average Metrics:
- CPU Time: 5751ms ± 801ms
- CPU Iterations: 76 ± 11
- CPU Time per Iteration: 75.66ms ± 1.75ms
- GPU Time: 35ms ± 4ms
- GPU Iterations: 61 ± 10
- GPU Time per Iteration: 0.58ms ± 0.02ms
- Average Speedup: 167x ± 36x

Key Observations:
1. GPU implementation shows dramatic performance improvement over previous version
2. Iteration counts are more consistent between CPU and GPU implementations
3. GPU time per iteration is extremely stable (very low standard deviation)
4. Both implementations achieve convergence reliably
5. Cluster distribution remains balanced (~20% per cluster) across runs
6. Per-iteration speedup has improved significantly from previous benchmarks

These example demonstrate:
- The GPU implementation achieving over 9x speedup over CPU
- GPU speedup is scaled with problem complexity
- Both implementations converging with well-defined clusters
- Even distribution of points across clusters
- ASCII visualization showing:
  - Four distinct clusters
  - Clear boundaries between clusters
  - Centroids ('C') positioned at cluster centers
  - Dense point distribution within each cluster
  - Numbers (0-3) representing points in different clusters

## Performance Considerations

- The GPU implementation is most effective for large datasets
- Performance scales with the number of points and dimensions
- Memory transfer overhead should be considered for smaller datasets
- Early convergence detection helps optimize iteration count
- Shared memory usage significantly improves performance

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
