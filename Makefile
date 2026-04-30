# Compiler
NVCC = nvcc

# Cluster-specific flags (for the 2080 Ti)
CLUSTER_FLAGS = -O3 -arch=sm_75

# Local-specific flags (automatically detects your local GPU architecture)
LOCAL_FLAGS = -O3 -arch=native

# Target names
TARGET = image_blur
SRC = image_blur.cu

# Default rule: builds for the cluster
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CLUSTER_FLAGS) $(SRC) -o $(TARGET)

# Local build: uses 'native' architecture detection
local: $(SRC)
	$(NVCC) $(LOCAL_FLAGS) $(SRC) -o $(TARGET)

# Clean rule
clean:
	rm -f $(TARGET) blurred.bmp

# Command to build and run locally with a test image
# Usage: make run_local IMG=my_photo.bmp
run_local: local
	./$(TARGET) $(IMG)

# Run on cluster (assumes you've already built with 'make')
run_cluster:
	./$(TARGET) $(IMG)

# alt local :  nvcc -allow-unsupported-compiler -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" blur.cu -o blur.exe