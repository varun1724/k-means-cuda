NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11

SRCS = main.cu kmeans_gpu.cu kmeans_cpu.cu
HEADERS = kmeans_gpu.h kmeans_cpu.h

OBJS = $(SRCS:.cu=.o)

all: kmeans

kmeans: $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(OBJS)

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) kmeans

.PHONY: all clean 