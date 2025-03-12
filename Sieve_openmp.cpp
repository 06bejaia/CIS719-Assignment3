#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>  // For sqrt()

void sieve(int n) {
    std::vector<bool> prime(n + 1, true);
    prime[0] = prime[1] = false;

    // Calculate square root of n
    int limit = std::sqrt(n);

    // Parallelized sieve using OpenMP
    #pragma omp parallel for
    for (int i = 2; i <= limit; i++) {
        if (prime[i]) {
            for (int j = i * i; j <= n; j += i) {
                prime[j] = false;
            }
        }
    }

    // Output the primes
    for (int i = 2; i <= n; i++) {
        if (prime[i]) std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main() {
    int n = 100;
    sieve(n);
    return 0;
}
