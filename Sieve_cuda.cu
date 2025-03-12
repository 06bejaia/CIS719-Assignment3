#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA Kernel for the sieve
__global__ void sieve_kernel(bool* prime, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate thread index
    if (i >= 2 && i <= n) {
        // Mark non-prime numbers by sieving
        for (int j = i * i; j <= n; j += i) {
            prime[j] = false;
        }
    }
}

// Function to perform sieve of Eratosthenes on the host
void sieve(int n) {
    bool* d_prime;
    bool* prime = new bool[n + 1]; // Allocate raw array for prime numbers
    std::fill(prime, prime + n + 1, true);  // Initialize all elements to true
    prime[0] = prime[1] = false;  // 0 and 1 are not primes

    // Allocate memory on the device
    cudaError_t err = cudaMalloc(&d_prime, (n + 1) * sizeof(bool));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        delete[] prime;  // Clean up host memory
        return;
    }

    // Copy initial data from host to device
    err = cudaMemcpy(d_prime, prime, (n + 1) * sizeof(bool), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        delete[] prime;
        cudaFree(d_prime);  // Clean up device memory
        return;
    }

    // Define block size and grid size
    int blockSize = 256;  // Number of threads per block
    int gridSize = (n + blockSize - 1) / blockSize;  // Number of blocks to cover 'n' elements

    // Launch the kernel to sieve primes
    sieve_kernel<<<gridSize, blockSize>>>(d_prime, n);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        delete[] prime;
        cudaFree(d_prime);  // Clean up device memory
        return;
    }

    // Copy result back from device to host
    err = cudaMemcpy(prime, d_prime, (n + 1) * sizeof(bool), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(err) << std::endl;
        delete[] prime;
        cudaFree(d_prime);  // Clean up device memory
        return;
    }

    // Free device memory
    cudaFree(d_prime);

    // Print primes
    for (int i = 2; i <= n; i++) {
        if (prime[i]) std::cout << i << " ";
    }
    std::cout << std::endl;

    // Clean up host memory
    delete[] prime;
}

int main() {
    sieve(100);  // Find primes up to 100
    return 0;
}
