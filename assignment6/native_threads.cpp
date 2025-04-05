#include <iostream>
#include <vector>
#include <thread>

const int N = 4;
std::vector<std::vector<int>> matrix = {{1, 2, 3, 4},
                                        {5, 6, 7, 8},
                                        {9, 10, 11, 12},
                                        {13, 14, 15, 16}};
std::vector<int> vec = {1, 2, 3, 4};
std::vector<int> result(N, 0);

void multiplyRow(int row) {
    for (int j = 0; j < N; j++) {
        result[row] += matrix[row][j] * vec[j];
    }
}

int main() {
    std::vector<std::thread> threads;
    for (int i = 0; i < N; i++) {
        threads.push_back(std::thread(multiplyRow, i));
    }
    for (auto &t : threads) {
        t.join();
    }

    std::cout << "Result: ";
    for (int r : result) std::cout << r << " ";
    std::cout << std::endl;
    return 0;
}
