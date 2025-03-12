# Compiler Flags
CXX = g++
NVCC = nvcc
MPICXX = mpic++
CXXFLAGS = -g -Wall -std=c++17
LDFLAGS = -pthread   # Ensure -pthread is used for linking
OMPFLAGS = -fopenmp
CUDAFLAGS = -arch=sm_60

# MPI-specific flags (adjusted based on your output)
MPI_INC_PATH = /usr/include/x86_64-linux-gnu/mpich
MPI_LIB_PATH = /usr/lib/x86_64-linux-gnu
MPI_LIBS = -lmpichcxx -lmpich

# Source Files
C_APPS = Sieve_threads Sieve_pthreads Sieve_mpi Sieve_openmp
CU_APPS = Sieve_cuda Sieve_thrust

# Compilation Rules
all: ${C_APPS} ${CU_APPS}

# C++ Native Threads (ensure linking with -pthread)
sieve_threads: Sieve_threads.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -pthread  # Added -pthread during linking

# POSIX Threads (pthreads)
sieve_pthreads: sieve_pthreads.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)  # $(LDFLAGS) contains -pthread for linking

# MPI
sieve_mpi: Sieve_mpi.cpp
	$(MPICXX) $(CXXFLAGS) -I$(MPI_INC_PATH) -L$(MPI_LIB_PATH) $(MPI_LIBS) -o $@ $<

# OpenMP
sieve_openmp: Sieve_openmp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $<

# CUDA
sieve_cuda: Sieve_cuda.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $<

# Thrust (Thrust runs on CUDA)
sieve_thrust: Sieve_thrust.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $<

# Clean Rule
clean:
	rm -f ${CU_APPS} ${C_APPS}
