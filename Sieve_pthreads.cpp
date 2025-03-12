#include <iostream>
#include <vector>
#include <pthread.h>
#include <cmath>

#define NUM_THREADS 4
int n = 100;
std::vector<bool> prime(n + 1, true);
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void* sieve_worker(void* arg) {
    int id = *(int*)arg;
    int start = (id * n / NUM_THREADS) + 2;
    int end = ((id + 1) * n / NUM_THREADS);
    
    for (int i = 2; i * i <= n; i++) {
        if (prime[i]) {
            pthread_mutex_lock(&mutex);
            for (int j = std::max(i * i, (start / i) * i); j <= end; j += i)
                prime[j] = false;
            pthread_mutex_unlock(&mutex);
        }
    }
    pthread_exit(nullptr);
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; ++i) {
        thread_ids[i] = i;
        pthread_create(&threads[i], nullptr, sieve_worker, &thread_ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; ++i)
        pthread_join(threads[i], nullptr);

    for (int i = 2; i <= n; i++)
        if (prime[i]) std::cout << i << " ";
    std::cout << std::endl;
    
    return 0;
}
