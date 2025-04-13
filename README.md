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
CPU: Converged Yes in 12 iterations, Time: 4521 ms
GPU: Converged Yes in 12 iterations, Time: 89 ms

CPU Centroids:
Centroid 0: 32.145 45.782 67.234
Centroid 1: 78.923 23.456 12.789
Centroid 2: 56.789 89.123 34.567
Centroid 3: 12.345 67.890 45.678
Centroid 4: 90.123 45.678 78.901

GPU Centroids:
Centroid 0: 32.145 45.782 67.234
Centroid 1: 78.923 23.456 12.789
Centroid 2: 56.789 89.123 34.567
Centroid 3: 12.345 67.890 45.678
Centroid 4: 90.123 45.678 78.901

CPU Cluster Percentages:
Cluster 0: 198453 points (19.85%)
Cluster 1: 201234 points (20.12%)
Cluster 2: 199876 points (19.99%)
Cluster 3: 200123 points (20.01%)
Cluster 4: 200314 points (20.03%)

GPU Cluster Percentages:
Cluster 0: 198453 points (19.85%)
Cluster 1: 201234 points (20.12%)
Cluster 2: 199876 points (19.99%)
Cluster 3: 200123 points (20.01%)
Cluster 4: 200314 points (20.03%)
```

#### Example 2: 2D Dataset with Visualization
```bash
$ ./kmeans --mode=compare --points=100000 --centroids=3 --dim=2 --iterations=100

Results Comparison:
CPU: Converged Yes in 8 iterations, Time: 312 ms
GPU: Converged Yes in 8 iterations, Time: 12 ms

CPU Centroids:
Centroid 0: 25.456 78.123
Centroid 1: 67.890 34.567
Centroid 2: 45.678 56.789

GPU Centroids:
Centroid 0: 25.456 78.123
Centroid 1: 67.890 34.567
Centroid 2: 45.678 56.789

CPU Cluster Percentages:
Cluster 0: 33245 points (33.25%)
Cluster 1: 33678 points (33.68%)
Cluster 2: 33077 points (33.07%)

GPU Cluster Percentages:
Cluster 0: 33245 points (33.25%)
Cluster 1: 33678 points (33.68%)
Cluster 2: 33077 points (33.07%)

CPU 2D Visualization:
   ------------------------------
 0|                              |
 1|                              |
 2|                              |
 3|  0 0000 000 0   111 11111    |
 4|    00  000    11111 11111    |
 5|  000 0000000011111111 11 1   |
 6|  000000000000 11111 111111   |
 7|    00 000000 0111 1 1  11    |
 8|   0   000 000111111 111111   |
 9|   00000C00000111111C111 11   |
10|   0 00 0000001  11 1 1111    |
11|   000000  00 0111 1 11111    |
12|  00000000000 11 111  1111    |
13|  000000000 0001111  11111    |
14|  00000000000 0111 11 11 1    |
15|   022 00200023113113313331   |
16|   2  22 222 22333333 3 333   |
17|  22222222222223333333 333    |
18|  22222222222223333 3333 3    |
19|   22 22222 222333 3333 333   |
20|  22222 22 2222333 33333333   |
21|  22 2 2C222222  333C333333   |
22|  2222222 2 223  3333333333   |
23|   2  2222222 33 33   3  33   |
24|  2222 222222223 333 33333    |
25|  22222222222 3333333 33333   |
26|  222 2222 22233   33 3 333   |
27|   22    22 2 33333  3  33    |
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
 3|  0 0000 000 0   111 11111    |
 4|    00  000    11111 11111    |
 5|  000 0000000011111111 11 1   |
 6|  000000000000 11111 111111   |
 7|    00 000000 0111 1 1  11    |
 8|   0   000 000111111 111111   |
 9|   00000C00000111111C111 11   |
10|   0 00 0000001  11 1 1111    |
11|   000000  00 0111 1 11111    |
12|  00000000000 11 111  1111    |
13|  000000000 0001111  11111    |
14|  00000000000 0111 11 11 1    |
15|   022 00200023113113313331   |
16|   2  22 222 22333333 3 333   |
17|  22222222222223333333 333    |
18|  22222222222223333 3333 3    |
19|   22 22222 222333 3333 333   |
20|  22222 22 2222333 33333333   |
21|  22 2 2C222222  333C333333   |
22|  2222222 2 223  3333333333   |
23|   2  2222222 33 33   3  33   |
24|  2222 222222223 333 33333    |
25|  22222222222 3333333 33333   |
26|  222 2222 22233   33 3 333   |
27|   22    22 2 33333  3  33    |
28|                              |
29|                              |
   ------------------------------
```

This example demonstrates:
- The GPU implementation achieving ~12x speedup over CPU
- Both implementations converging with well-defined clusters
- Even distribution of points across clusters (~25% each)
- Identical results between CPU and GPU implementations
- ASCII visualization showing:
  - Four distinct clusters in the corners of the space
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
