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
./kmeans --mode=compare --points=10000 --centroids=4 --dim=2 --iterations=100
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
