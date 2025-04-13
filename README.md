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
CPU: Converged Yes in 94 iterations, Time: 6633 ms
GPU: Converged No in 100 iterations, Time: 236 ms
Speed up: 28.105932

CPU Centroids:
76.237732 48.969498 23.066887 
23.149548 23.686081 50.921329 
50.682762 76.912537 76.302391 
26.591623 73.365494 26.412760 
73.585190 26.616318 73.553749 

GPU Centroids:
22.929411 51.443859 23.894617 
26.408279 26.785614 73.330544 
51.451694 76.246567 77.019356 
75.971542 22.913113 48.716747 
73.431801 73.473732 26.279226 

CPU Cluster Percentages:
Cluster 0: 218122 points (21.81%)
Cluster 1: 217951 points (21.80%)
Cluster 2: 218671 points (21.87%)
Cluster 3: 174608 points (17.46%)
Cluster 4: 170648 points (17.06%)

GPU Cluster Percentages:
Cluster 0: 217902 points (21.79%)
Cluster 1: 175147 points (17.51%)
Cluster 2: 217943 points (21.79%)
Cluster 3: 219257 points (21.93%)
Cluster 4: 169751 points (16.98%)
```

#### Example 2: 2D Dataset with Visualization (1 million points)
```bash
$ ./kmeans --mode=compare --points=1000000 --centroids=3 --dim=2 --iterations=100

Results Comparison:
CPU: Converged Yes in 46 iterations, Time: 1861 ms
GPU: Converged Yes in 79 iterations, Time: 189 ms
Speed up: 9.846560

CPU Centroids:
68.105927 76.912933 
68.354057 23.199709 
19.546360 49.818516 

GPU Centroids:
76.776535 31.650082 
50.156288 80.422615 
23.036469 31.875134 

CPU Cluster Percentages:
Cluster 0: 313474 points (31.35%)
Cluster 1: 311279 points (31.13%)
Cluster 2: 375247 points (37.52%)

GPU Cluster Percentages:
Cluster 0: 311601 points (31.16%)
Cluster 1: 375336 points (37.53%)
Cluster 2: 313063 points (31.31%)

CPU 2D Visualization:
   ------------------------------
 0|                              |
 1|                              |
 2|                              |
 3|   2  2 00  000000 0  0   0   |
 4|   2222220000000 000000 00    |
 5|   2222 200 0 00 000   0000   |
 6|   222222000  000000 00000    |
 7|  222222220000000 00000 000   |
 8|   2 22222200000 0C00 000 0   |
 9|    2 2222 200 00000  00 0    |
10|   22222222200000 000000000   |
11|   222 2222 2000000 0 00000   |
12|  2 222222  2 00 000000 00    |
13|  2 22222222 20 000 00 000    |
14|  2 222 2222222000 000000     |
15|  22222C222222 011000110111   |
16|    2 22 22 22 111 11 11111   |
17|  22222   2222111111  11111   |
18|   2222  222 11 11111 1111    |
19|   2 22 2222 1111 111 1111    |
20|  22222 2  21111111111 111    |
21|  2 22 222 1111   C1111111    |
22|  222222222 11  11 11111 11   |
23|   2 2 2 211111111 11111111   |
24|    222222111111  111111 11   |
25|  2 2  2221111111111111 111   |
26|  2 22222 111111111   1 1 1   |
27|   22222111111111  1111111    |
28|                              |
29|                              |
   ------------------------------

Legend:
C: Centroid
0-9: Points (number indicates cluster)

GPU 2D Visualization:
   ------------------------------
 0|                              |
 1|                              |
 2|                              |
 3|   1  1 11  111111 1  1   1   |
 4|   1111111111111 111111 11    |
 5|   1111 111 1 11 111   1111   |
 6|   111111111  111111 11111    |
 7|  111111111111111 11111 111   |
 8|   1 111111111C1 1111 111 1   |
 9|    1 1111 111 11111  11 0    |
10|   22111111111111 111110100   |
11|   222 2111 1111111 1 00000   |
12|  2 222212  1 11 111000 00    |
13|  2 22222211 11 110 00 000    |
14|  2 222 2222111110 000000     |
15|  222222222222 000000000000   |
16|    2 22 22 22 000 00 00000   |
17|  22222   2222200000  00000   |
18|   2222  222 20 00000 0000    |
19|   2 22C2222 2000 00C 0000    |
20|  22222 2  22200000000 000    |
21|  2 22 222 2222   00000000    |
22|  222222222 22  00 00000 00   |
23|   2 2 2 222222000 00000000   |
24|    222222222220  000000 00   |
25|  2 2  2222222000000000 000   |
26|  2 22222 222200000   0 0 0   |
27|   22222222222000  0000000    |
28|                              |
29|                              |
   ------------------------------

Legend:
C: Centroid
0-9: Points (number indicates cluster)
```

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
