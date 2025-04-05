#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

const int N = 4;
std::vector<int> result(N, 0);

int main() {
    std::vector<int> h_matrix_flat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int> h_vec = {1, 2, 3, 4};

    thrust::device_vector<int> d_matrix(h_matrix_flat.begin(), h_matrix_flat.end());
    thrust::device_vector<int> d_vec(h_vec.begin(), h_vec.end());
    thrust::device_vector<int> d_result(N, 0);

    thrust::transform(d_matrix.begin(), d_matrix.end(), d_vec.begin(), d_result.begin(), thrust::multiplies<int>());

    thrust::copy(d_result.begin(), d_result.end(), result.begin());

    std::cout << "Result: ";
    for (int r : result) std::cout << r << " ";
    std::cout << std::endl;
    return 0;
}
