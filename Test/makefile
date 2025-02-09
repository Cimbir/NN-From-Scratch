export PATH := /opt/cuda/bin:$(PATH)
export LD_LIBRARY_PATH := /opt/cuda/lib64:$(LD_LIBRARY_PATH)

NVCC := /opt/cuda/bin/nvcc
SRC := gpu_usage.cu
OBJ := $(SRC:.cu=.o)
EXEC := program

all: $(EXEC)

$(EXEC): $(OBJ)
	$(NVCC) -o $@ $^ -lcudart

%.o: %.cu
	$(NVCC) -c $< -o $@ -lcudart
	
clean:
	rm -f $(EXEC)