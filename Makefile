NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_60

TARGET = kmeans
SRCS = main.cu kmeans.cu
OBJS = $(SRCS:.cu=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean 