#ifndef UTIL_H
#define UTIL_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__device__ inline void
__print_mat(int M, int N, float *mat)
{
    printf("%dx%d:\n[\n", M, N);
    for(int i=0; i<M; ++i){
        printf("[ ");
        for(int j=0; j<N; ++j){
            printf("%7.4f, ", mat[i * N + j]);
        }
        printf("],\n");
    }
    printf("]\n");
}

// https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#endif
