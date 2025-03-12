#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <iostream>
#include <cmath>

// Kernel function to mark multiples of a prime
struct MarkMultiples {
    int prime;
    MarkMultiples(int p) : prime(p) {}

    __device__ bool operator()(const int& x) const {
        return (x > prime) && (x % prime == 0); // Remove multiples of prime
    }
};

// GPU-accelerated Sieve of Eratosthenes
void sieve(int n) {
    thrust::device_vector<int> numbers(n - 1);
    thrust::sequence(numbers.begin(), numbers.end(), 2); // Fill with numbers 2 to n

    int sqrt_n = std::sqrt(n);
    
    for (int i = 2; i <= sqrt_n; ++i) {
        thrust::device_vector<int>::iterator new_end;
        new_end = thrust::remove_if(thrust::device, numbers.begin(), numbers.end(), MarkMultiples(i));
        numbers.resize(new_end - numbers.begin()); // Resize to remove non-prime numbers
    }

    // Copy primes back to CPU
    std::vector<int> primes(numbers.size());
    thrust::copy(numbers.begin(), numbers.end(), primes.begin());

    // Print results
    for (int p : primes) std::cout << p << " ";
    std::cout << std::endl;
}

int main() {
    int n = 100;
    sieve(n);
    return 0;
}

