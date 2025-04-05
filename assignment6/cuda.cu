#include <iostream>
#include <vector>
#include <cuda_runtime.h>

const int N = 4;
std::vector<int> result(N, 0);

__global__ void cuda_kernel(int* d_matrix, int* d_vec, int* d_result, int n) {
    int i = threadIdx.x;
    if (i < n) {
        for (int j = 0; j < n; j++) {
            d_result[i] += d_matrix[i * n + j] * d_vec[j];
        }
    }
}

int main() {
    int h_matrix[N][N] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
    int h_vec[N] = {1, 2, 3, 4};
    int *d_matrix, *d_vec, *d_result;

    cudaMalloc(&d_matrix, N * N * sizeof(int));
    cudaMalloc(&d_vec, N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(int));

    cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, h_vec, N * sizeof(int), cudaMemcpyHostToDevice);

    cuda_kernel<<<1, N>>>(d_matrix, d_vec, d_result, N);

    cudaMemcpy(result.data(), d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_vec);
    cudaFree(d_result);

    std::cout << "Result: ";
    for (int r : result) std::cout << r << " ";
    std::cout << std::endl;
    return 0;
}
