#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

// Parallel Sieve of Eratosthenes
void sieve(int n, int rank, int size) {
    int sqrt_n = std::sqrt(n);  
    std::vector<char> is_prime(n + 1, 1);  // Use char instead of bool for MPI compatibility

    if (rank == 0) {
        is_prime[0] = is_prime[1] = 0;  
        for (int i = 2; i * i <= n; i++) {
            if (is_prime[i]) {
                for (int j = i * i; j <= n; j += i) {
                    is_prime[j] = 0;
                }
            }
        }
    }

    // Broadcast using MPI_CHAR
    MPI_Bcast(is_prime.data(), n + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Define local range for each process
    int start = rank * (n / size) + 1;
    int end = (rank + 1) * (n / size);
    if (start < 2) start = 2;

    std::vector<char> local_prime(end - start + 1, 1);

    // Mark non-primes using small primes from sqrt(n)
    for (int i = 2; i <= sqrt_n; i++) {
        if (is_prime[i]) {
            int first_multiple = std::max(i * i, (start + i - 1) / i * i);
            for (int j = first_multiple; j <= end; j += i) {
                local_prime[j - start] = 0;
            }
        }
    }

    // Gather and print results at rank 0
    if (rank == 0) {
        for (int i = 2; i <= sqrt_n; i++) {
            if (is_prime[i]) std::cout << i << " ";
        }
        for (int i = start; i <= end; i++) {
            if (local_prime[i - start]) std::cout << i << " ";
        }
        std::cout << std::endl;
    } else {
        for (int i = start; i <= end; i++) {
            if (local_prime[i - start]) {
                std::cout << i << " ";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int rank, size, n = 100;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sieve(n, rank, size);

    MPI_Finalize();
    return 0;
}
