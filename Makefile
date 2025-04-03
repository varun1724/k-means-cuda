NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60 -std=c++11
INCLUDES = -I./include

# Source files
SRCS = src/main.cu src/kmeans_cpu.cu
HEADERS = include/kmeans_cpu.h include/kmeans_gpu.h

# Object files
OBJS = $(SRCS:.cu=.o)

# Executable
TARGET = kmeans

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean 