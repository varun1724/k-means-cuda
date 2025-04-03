# CUDA-Accelerated K-Means Clustering

This project implements k-means clustering algorithm with both CPU and GPU (CUDA) implementations. The goal is to demonstrate the performance benefits of GPU acceleration for this common machine learning algorithm.

## Project Structure

```
k-means-cuda/
├── .gitignore           # Git ignore file for build artifacts and system files
├── Makefile            # Build configuration
├── main.cu             # Main program entry point
├── kmeans_cpu.h        # CPU implementation header
├── kmeans_cpu.cu       # CPU implementation
├── kmeans_gpu.h        # GPU implementation header
├── kmeans_gpu.cu       # GPU implementation (to be implemented)
└── performance_test.cu # Performance testing and comparison
```

## Features

- CPU implementation of k-means clustering
- GPU implementation using CUDA (to be implemented)
- Performance testing framework
- Configurable parameters for testing different dataset sizes
- Detailed result verification and debugging output

## Requirements

- CUDA Toolkit (version 11.0 or higher)
- C++11 compatible compiler
- Make build system

## Building

To build the project:

```bash
make
```

To clean build artifacts:

```bash
make clean
```

## Running

To run the performance test:

```bash
./kmeans_test
```

## Implementation Details

### CPU Implementation
The CPU implementation uses a straightforward approach with:
- Sequential point assignment
- Centroid updates using running sums
- Early convergence detection

### GPU Implementation (To Be Implemented)
The GPU implementation will utilize:
- Parallel point assignment
- Shared memory optimizations
- Efficient centroid updates
- Memory coalescing for better performance

## Performance Testing

The performance test framework:
1. Generates test data with controlled properties
2. Runs both CPU and GPU implementations
3. Measures execution time
4. Verifies result correctness
5. Provides detailed debugging information

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
