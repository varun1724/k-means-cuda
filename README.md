# K-Means Clustering Implementation

This project implements the K-Means clustering algorithm with both CPU and GPU (CUDA) implementations. The project is structured to provide a clean separation between CPU and GPU code, making it easy to compare performance and maintain the codebase.

## Project Structure

```
k-means-cuda/
├── include/              # Header files
│   ├── kmeans_cpu.h     # CPU implementation declarations
│   └── kmeans_gpu.h     # GPU implementation declarations (for future use)
├── src/                 # Source files
│   ├── main.cu          # Main program with all modes
│   ├── kmeans_cpu.cu    # CPU implementation
│   └── kmeans_gpu.cu    # GPU implementation (to be implemented)
├── Makefile            # Build configuration
└── README.md           # This file
```

## Features

- **CPU Implementation**: Standard implementation of K-Means clustering
- **Multiple Operation Modes**: Simple, Test, and Verify modes for different use cases
- **Configurable Parameters**: Adjustable number of points, centroids, dimensions, and iterations
- **Built-in Verification**: Comprehensive clustering quality metrics
- **Future GPU Support**: Prepared for CUDA implementation

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
   make
   ```

3. Clean build files:
   ```bash
   make clean
   ```

## Usage

The program provides three modes of operation, all accessible through a single executable:

### Simple Mode
Displays only the final centroids:
```bash
./kmeans --mode=simple [options]
```

### Test Mode
Shows detailed information including timing and cluster assignments:
```bash
./kmeans --mode=test [options]
```

### Verify Mode
Performs comprehensive verification and displays detailed metrics:
```bash
./kmeans --mode=verify [options]
```

### Available Options

- `--points=N`: Number of points (default: 1000)
- `--centroids=N`: Number of centroids (default: 2)
- `--dim=N`: Number of dimensions (default: 2)
- `--iterations=N`: Maximum iterations (default: 10)
- `--mode=MODE`: Operation mode (simple/test/verify)
- `--help`: Show help message

### Examples

1. Basic clustering with default parameters:
   ```bash
   ./kmeans --mode=simple
   ```

2. Detailed testing with custom parameters:
   ```bash
   ./kmeans --mode=test --points=10000 --centroids=5 --iterations=100
   ```

3. Comprehensive verification:
   ```bash
   ./kmeans --mode=verify --points=10000 --centroids=4 --iterations=100
   ```

## Output Information

### Simple Mode
- Final centroid coordinates

### Test Mode
- Execution time
- Final centroid coordinates
- First 10 point assignments
- Convergence information

### Verify Mode
- Execution time
- Total within-cluster sum of squares
- Average distance to centroid
- Cluster size distribution
- Convergence stability check
- Empty cluster detection

## Performance Considerations

- The implementation uses efficient data structures and algorithms
- Early convergence detection based on centroid movement
- Configurable convergence tolerance
- Future GPU implementation will provide acceleration for large datasets

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
