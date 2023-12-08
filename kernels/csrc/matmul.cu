#include "matmul.h"
#include "util.cuh"
#include <stdio.h>

#define Br 32
#define Bc 32
#define Bk 32

__global__ void
__simple_matmul(
    int M, int N, int K,
    float *C, int C_strideM, int C_strideN,
    float *A, int A_strideM, int A_strideK,
    float *B, int B_strideK, int B_strideN
)
{
    for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<M; i += gridDim.x*blockDim.x) {
        for(int j=blockIdx.y*blockDim.y+threadIdx.y; j<N; j += gridDim.y*blockDim.y) {
            float dot = 0;
            for(int k=0; k<K; ++k) {
                dot += A[i * A_strideM + k * A_strideK] * B[k * B_strideK + j * B_strideN];
            }
            C[i * C_strideM + j * C_strideN] += dot;
        }
    }
}

__global__ void
__tiled_matmul(
    int M, int N, int K,
    float *C, int C_strideM, int C_strideN,
    float *A, int A_strideM, int A_strideK,
    float *B, int B_strideK, int B_strideN
)
{
    __shared__ float C_tile[Br][Bc];
    __shared__ float A_tile[Br][Bk];
    __shared__ float B_tile[Bk][Bc];

    for(int i=blockIdx.x*Br; i<M; i += gridDim.x*Br) {
        for(int j=blockIdx.y*Bc; j<N; j += gridDim.y*Bc) {
            const int Br_ = min(Br, M-i);
            const int Bc_ = min(Bc, N-j);
            // load C
            for(int ci=threadIdx.x; ci<Br_; ci += blockDim.x) {
                for(int cj=threadIdx.y; cj<Bc_; cj += blockDim.y) {
                    C_tile[ci][cj] = C[(i + ci) * C_strideM + (j + cj) * C_strideN];
                }
            }
            for(int k=0; k<K; k += Bk) {
                const int Bk_ = min(Bk, K-k);
                // load A, B
                for(int ai=threadIdx.x; ai<Br_; ai += blockDim.x) {
                    for(int ak=threadIdx.y; ak<Bk_; ak += blockDim.y) {
                        A_tile[ai][ak] = A[(i + ai) * A_strideM + (k + ak) * A_strideK];
                    }
                }
                for(int bk=threadIdx.x; bk<Bk_; bk += blockDim.x) {
                    for(int bj=threadIdx.y; bj<Bc_; bj += blockDim.y) {
                        B_tile[bk][bj] = B[(k + bk) * B_strideK + (j + bj) * B_strideN];
                    }
                }
                __syncthreads();
                // compute C tile
                for(int ci=threadIdx.x; ci<Br_; ci += blockDim.x) {
                    for(int cj=threadIdx.y; cj<Bc_; cj += blockDim.y) {
                        for(int ck=0; ck<Bk_; ++ck) {
                            C_tile[ci][cj] += A_tile[ci][ck] * B_tile[ck][cj];
                        }
                    }
                }
                __syncthreads();
            }
            // write C
            for(int ci=threadIdx.x; ci<Br_; ci += blockDim.x) {
                for(int cj=threadIdx.y; cj<Bc_; cj += blockDim.y) {
                    C[(i + ci) * C_strideM + (j + cj) * C_strideN] += C_tile[ci][cj];
                }
            }
        }
    }
}

void
simple_matmul(
    torch::Tensor C,
    torch::Tensor A,
    torch::Tensor B
)
{
    int M = C.size(0);
    int N = C.size(1);
    int K = A.size(1);

    int C_strideM = C.stride(0);
    int C_strideN = C.stride(1);
    int A_strideM = A.stride(0);
    int A_strideK = A.stride(1);
    int B_strideK = B.stride(0);
    int B_strideN = B.stride(1);

    float *C_ptr = C.data_ptr<float>();
    float *A_ptr = A.data_ptr<float>();
    float *B_ptr = B.data_ptr<float>();

    int block_size = 32;
    dim3 grid(
        (M + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );
    dim3 block(block_size, block_size);
    __simple_matmul<<<grid, block>>>(
        M, N, K,
        C_ptr, C_strideM, C_strideN,
        A_ptr, A_strideM, A_strideK,
        B_ptr, B_strideK, B_strideN
    );
}

void
tiled_matmul(
    torch::Tensor C,
    torch::Tensor A,
    torch::Tensor B
)
{
    int M = C.size(0);
    int N = C.size(1);
    int K = A.size(1);

    int C_strideM = C.stride(0);
    int C_strideN = C.stride(1);
    int A_strideM = A.stride(0);
    int A_strideK = A.stride(1);
    int B_strideK = B.stride(0);
    int B_strideN = B.stride(1);

    float *C_ptr = C.data_ptr<float>();
    float *A_ptr = A.data_ptr<float>();
    float *B_ptr = B.data_ptr<float>();

    int block_size = 32;
    dim3 grid(
        (M + block_size - 1) / block_size,
        (N + block_size - 1) / block_size
    );
    dim3 block(block_size, block_size);
    __tiled_matmul<<<grid, block>>>(
        M, N, K,
        C_ptr, C_strideM, C_strideN,
        A_ptr, A_strideM, A_strideK,
        B_ptr, B_strideK, B_strideN
    );
}
