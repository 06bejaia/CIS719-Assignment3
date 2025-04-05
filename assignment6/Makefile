# Compiler
CXX = g++
NVCC = nvcc

# Compiler flags
CXXFLAGS = -O2 -Wall -fopenmp
MPIFLAGS = -lmpi
CUDAFLAGS = -arch=sm_50

# Executable names
EXEC_NATIVE = native_threads
EXEC_PTHREADS = posix_threads
EXEC_OPENMP = openmp
EXEC_MPI = mpi
EXEC_CUDA = cuda
EXEC_THRUST = thrust

# Source files
SRC_NATIVE = native_threads.cpp
SRC_PTHREADS = posix_threads.cpp
SRC_OPENMP = openmp.cpp
SRC_MPI = mpi.cpp
SRC_CUDA = cuda.cu
SRC_THRUST = thrust.cu

# Default target
all: $(EXEC_NATIVE) $(EXEC_PTHREADS) $(EXEC_OPENMP) $(EXEC_MPI) $(EXEC_CUDA) $(EXEC_THRUST)

# Build rules
$(EXEC_NATIVE): $(SRC_NATIVE)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(EXEC_PTHREADS): $(SRC_PTHREADS)
	$(CXX) $(CXXFLAGS) -pthread -o $@ $^

$(EXEC_OPENMP): $(SRC_OPENMP)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(EXEC_MPI): $(SRC_MPI)
	$(CXX) $(CXXFLAGS) $(MPIFLAGS) -o $@ $^

$(EXEC_CUDA): $(SRC_CUDA)
	$(NVCC) $(CUDAFLAGS) -o $@ $^

$(EXEC_THRUST): $(SRC_THRUST)
	$(NVCC) $(CUDAFLAGS) -o $@ $^

# Run all
run: all
	./$(EXEC_NATIVE)
	./$(EXEC_PTHREADS)
	./$(EXEC_OPENMP)
	mpirun -np 4 ./$(EXEC_MPI)
	./$(EXEC_CUDA)
	./$(EXEC_THRUST)

# Clean up binaries
clean:
	rm -f $(EXEC_NATIVE) $(EXEC_PTHREADS) $(EXEC_OPENMP) $(EXEC_MPI) $(EXEC_CUDA) $(EXEC_THRUST)
