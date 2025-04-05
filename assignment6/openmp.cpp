#include <iostream>
#include <vector>
#include <omp.h>

const int N = 4;
std::vector<std::vector<int>> matrix = {{1, 2, 3, 4},
                                        {5, 6, 7, 8},
                                        {9, 10, 11, 12},
                                        {13, 14, 15, 16}};
std::vector<int> vec = {1, 2, 3, 4};
std::vector<int> result(N, 0);

int main() {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i][j] * vec[j];
        }
    }

    std::cout << "Result: ";
    for (int r : result) std::cout << r << " ";
    std::cout << std::endl;
    return 0;
}
