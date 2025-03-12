#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <mutex>

// Mutex to prevent race conditions
std::mutex mtx;

// Worker function to mark non-primes in a given range
void sieve_worker(std::vector<bool>& prime, int start, int end, int sqrt_n) {
    for (int i = 2; i <= sqrt_n; i++) {
        if (prime[i]) {
            // Find the first multiple of i in range [start, end]
            int first_multiple = std::max(i * i, (start + i - 1) / i * i);
            for (int j = first_multiple; j <= end; j += i) {
                prime[j] = false;
            }
        }
    }
}

// Multi-threaded Sieve of Eratosthenes
void sieve(int n, int num_threads) {
    std::vector<bool> prime(n + 1, true);
    prime[0] = prime[1] = false;

    int sqrt_n = std::sqrt(n);
    std::vector<std::thread> threads;
    int chunk_size = (n - sqrt_n) / num_threads;

    // Mark non-primes up to sqrt(n) in the main thread
    for (int i = 2; i <= sqrt_n; i++) {
        if (prime[i]) {
            for (int j = i * i; j <= sqrt_n; j += i) {
                prime[j] = false;
            }
        }
    }

    // Launch worker threads for numbers greater than sqrt(n)
    for (int i = 0; i < num_threads; ++i) {
        int start = sqrt_n + (i * chunk_size) + 1;
        int end = (i == num_threads - 1) ? n : (start + chunk_size - 1);
        threads.emplace_back(sieve_worker, std::ref(prime), start, end, sqrt_n);
    }

    for (auto& t : threads) t.join();

    // Print prime numbers
    for (int i = 2; i <= n; ++i) {
        if (prime[i]) std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main() {
    int n = 100;
    int num_threads = 4;
    sieve(n, num_threads);
    return 0;
}
