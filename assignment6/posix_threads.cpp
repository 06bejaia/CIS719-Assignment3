#include <iostream>
#include <vector>
#include <pthread.h>

const int N = 4;
std::vector<std::vector<int>> matrix = {{1, 2, 3, 4},
                                        {5, 6, 7, 8},
                                        {9, 10, 11, 12},
                                        {13, 14, 15, 16}};
std::vector<int> vec = {1, 2, 3, 4};
std::vector<int> result(N, 0);
pthread_mutex_t mutex;

void* pthread_worker(void* arg) {
    int row = *(int*)arg;
    for (int j = 0; j < N; j++) {
        pthread_mutex_lock(&mutex);
        result[row] += matrix[row][j] * vec[j];
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    pthread_t threads[N];
    int indices[N];
    pthread_mutex_init(&mutex, NULL);
    for (int i = 0; i < N; i++) {
        indices[i] = i;
        pthread_create(&threads[i], NULL, pthread_worker, &indices[i]);
    }
    for (int i = 0; i < N; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&mutex);

    std::cout << "Result: ";
    for (int r : result) std::cout << r << " ";
    std::cout << std::endl;
    return 0;
}
