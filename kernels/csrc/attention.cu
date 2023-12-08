#include "attention.h"
#include "math.cuh"
#include "util.cuh"
#include <stdio.h>
#include <torch/torch.h>

#define Br 32
#define Bc 32 
#define D_ 32

__global__ void
__simple_self_attention(
    int B, int H, int N, int D,
    float *O, int O_strideN, int O_strideD,
    float *Q, int Q_strideN, int Q_strideD,
    float *K, int K_strideN, int K_strideD,
    float *V, int V_strideN, int V_strideD,
    float *L,
    float *M
)
{
    __shared__ float Oi[Br][D_];
    __shared__ float Qi[Br][D_];
    __shared__ float Kj[Bc][D_];
    __shared__ float Vj[Bc][D_];
    __shared__ float Sij[Br][Bc];
    __shared__ float Pij[Br][Bc];
    __shared__ float Li[Br];
    __shared__ float Lij[Br];
    __shared__ float Li_new[Br];
    __shared__ float Mi[Br];
    __shared__ float Mij[Br];
    __shared__ float Mi_new[Br];
    __shared__ float PV[Br][D_];
    // get local bufs for this block's batch and head
    float *O_ = &O[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *Q_ = &Q[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *K_ = &K[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *V_ = &V[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *L_ = &L[blockIdx.x * H*N + blockIdx.y * N];
    float *M_ = &M[blockIdx.x * H*N + blockIdx.y * N];

    for(int j=0; j<N; j += Bc) {
        const int Bc_ = min(Bc, N-j);
        // load Kj to SRAM
        for(int ki=threadIdx.x; ki<Bc_; ki += blockDim.x) {
            for(int kj=threadIdx.y; kj<D; kj += blockDim.y) {
                // printf("loading K_[(%d + %d) * %d + %d * %d] = K_[%d] = %f into Kj[%d][%d]\n",
                //     j, ki, K_strideN, kj, K_strideD, (j + ki) * K_strideN + kj * K_strideD, K_[(j + ki) * K_strideN + kj * K_strideD], ki, kj
                // );
                Kj[ki][kj] = K_[(j + ki) * K_strideN + kj * K_strideD];
            }
        }
        // load Vj to SRAM
        for(int vi=threadIdx.x; vi<Bc_; vi += blockDim.x) {
            for(int vj=threadIdx.y; vj<D; vj += blockDim.y) {
                Vj[vi][vj] = V_[(j + vi) * V_strideN + vj * V_strideD];
            }
        }
        for(int i=blockIdx.z*Br; i<N; i += gridDim.z*Br) {
            const int Br_ = min(Br, N-i);
            // load Qi to SRAM
            for(int qi=threadIdx.x; qi<Br_; qi += blockDim.x) {
                for(int qj=threadIdx.y; qj<D; qj += blockDim.y) {
                    Qi[qi][qj] = Q_[(i + qi) * Q_strideN + qj * Q_strideD];
                }
            }
            // load Oi to SRAM
            for(int oi=threadIdx.x; oi<Br_; oi += blockDim.x) {
                for(int oj=threadIdx.y; oj<D; oj += blockDim.y) {
                    Oi[oi][oj] = O_[(i + oi) * O_strideN + oj * O_strideD];
                }
            }
            // load Li to SRAM
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                Li[li] = L_[i + li];
            }
            // load Mi to SRAM
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                Mi[mi] = M_[i + mi];
            }
            __syncthreads();
            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            //     printf("Kj=%d:\n", j);
            //     __print_mat(Bc, D, (float *)Kj);
            //     printf("Vj=%d:\n", j);
            //     __print_mat(Bc, D, (float *)Vj);
            //     printf("Qi=%d:\n", i);
            //     __print_mat(Br, D, (float *)Qi);
            //     printf("Oi=%d:\n", i);
            //     __print_mat(Br, D, (float *)Oi);
            //     printf("Li=%d:\n", i);
            //     __print_mat(Br, 1, (float *)Li);
            //     printf("Mi=%d:\n", i);
            //     __print_mat(Br, 1, (float *)Mi);
            // }
            // compute Sij
            for(int si=threadIdx.x; si<Br_; si += blockDim.x) {
                for(int sj=threadIdx.y; sj<Bc_; sj += blockDim.y) {
                    float dot = 0;
                    for(int sk=0; sk<D; ++sk) {
                        dot += Qi[si][sk] * Kj[sj][sk];
                    }
                    Sij[si][sj] = dot / sqrt(D);
                }
            }
            __syncthreads();
            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            //     printf("Si=%d,j=%d:\n", i, j);
            //     __print_mat(Br, Bc, (float *)Sij);
            // }
            // compute Mij
            for(int si=threadIdx.x; si<Br_; si += blockDim.x) {
                if(threadIdx.y == 0) {
                    float rowmax = Sij[si][0];
                    for(int sj=0; sj<Bc_; ++sj) {
                        rowmax = max(rowmax, Sij[si][sj]);
                    }
                    Mij[si] = rowmax;
                }
            }
            __syncthreads();
            // compute Pij
            for(int si=threadIdx.x; si<Br_; si += blockDim.x) {
                for(int sj=threadIdx.y; sj<Bc_; sj += blockDim.y) {
                    Pij[si][sj] = exp(Sij[si][sj] - Mij[si]);
                }
            }
            // compute Lij
            for(int pi=threadIdx.x; pi<Br_; pi += blockDim.x) {
                if(threadIdx.y == 0) {
                    float rowsum = 0;
                    for(int pj=0; pj<Bc_; ++pj) {
                        rowsum += Pij[pi][pj];
                    }
                    Lij[pi] = rowsum;
                }
            }
            // compute Mi_new
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                Mi_new[mi] = max(Mi[mi], Mij[mi]);
            }
            // compute Li_new
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                Li_new[li] = exp(Mi[li] - Mi_new[li]) * Li[li] + exp(Mij[li] - Mi_new[li]) * Lij[li];
            }
            __syncthreads();
            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            //     printf("Mi=%d,j=%d:\n", i, j);
            //     __print_mat(Br, 1, (float *)Mij);
            //     printf("Pi=%d,j=%d:\n", i, j);
            //     __print_mat(Br, Bc, (float *)Pij);
            //     printf("Li=%d,j=%d:\n", i, j);
            //     __print_mat(Br, 1, (float *)Lij);
            //     printf("Mi=%d_new:\n", i);
            //     __print_mat(Br, 1, (float *)Mi_new);
            //     printf("Li=%d_new:\n", i);
            //     __print_mat(Br, 1, (float *)Li_new);
            // }
            // compute PV
            for(int ai=threadIdx.x; ai<Br_; ai += blockDim.x) {
                for(int aj=threadIdx.y; aj<D; aj += blockDim.y) {
                    float dot = 0;
                    for(int ak=0; ak<Bc_; ++ak) {
                        dot += Pij[ai][ak] * Vj[ak][aj];
                    }
                    PV[ai][aj] = dot;
                }
            }
            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            //     printf("PVi=%d,j=%d:\n", i, j);
            //     __print_mat(Br, D, (float *)PV);
            // }
            // update Oi
            for(int oi=threadIdx.x; oi<Br_; oi += blockDim.x) {
                for(int oj=threadIdx.y; oj<D; oj += blockDim.y) {
                    Oi[oi][oj] = (Li[oi] * exp(Mi[oi]  - Mi_new[oi]) * Oi[oi][oj]
                                         + exp(Mij[oi] - Mi_new[oi]) * PV[oi][oj])
                                 / Li_new[oi];
                }
            }
            // update Li
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                Li[li] = Li_new[li];
            }
            // update Mi
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                Mi[mi] = Mi_new[mi];
            }
            // write O to HBM
            for(int oi=threadIdx.x; oi<Br_; oi += blockDim.x) {
                for(int oj=threadIdx.y; oj<D; oj += blockDim.y) {
                    O_[(i + oi) * O_strideN + oj * O_strideD] = Oi[oi][oj];
                }
            }
            // write L to HBM
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                L_[i + li] = Li[li];
            }
            // write M to HBM
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                M_[i + mi] = Mi[mi];
            }
            __syncthreads();
            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            //     printf("updated Oi=%d:\n", i);
            //     __print_mat(Br, D, (float *)Oi);
            //     printf("updated Li=%d:\n", i);
            //     __print_mat(Br, 1, (float *)Li);
            //     printf("updated Mi=%d:\n", i);
            //     __print_mat(Br, 1, (float *)Mi);
            // }
        }       
    }
}

torch::Tensor
simple_self_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    torch::Tensor O = torch::zeros_like(Q);
    torch::Tensor L = torch::zeros({B, H, N}, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor M = torch::zeros({B, H, N}, torch::TensorOptions().device(torch::kCUDA));

    int O_strideN = O.stride(2);
    int O_strideD = O.stride(3);
    int Q_strideN = Q.stride(2);
    int Q_strideD = Q.stride(3);
    int K_strideN = K.stride(2);
    int K_strideD = K.stride(3);
    int V_strideN = V.stride(2);
    int V_strideD = V.stride(3);

    float *O_ptr = O.data_ptr<float>();
    float *Q_ptr = Q.data_ptr<float>();
    float *K_ptr = K.data_ptr<float>();
    float *V_ptr = V.data_ptr<float>();
    float *L_ptr = L.data_ptr<float>();
    float *M_ptr = M.data_ptr<float>();

    dim3 grid(
        B,
        H,
        (N + Br - 1) / Br
    );
    dim3 block(Br, D);
    __simple_self_attention<<<grid, block>>>(
        B, H, N, D,
        O_ptr, O_strideN, O_strideD,
        Q_ptr, Q_strideN, Q_strideD,
        K_ptr, K_strideN, K_strideD,
        V_ptr, V_strideN, V_strideD,
        L_ptr,
        M_ptr
    );
    return O;
}

template <int THREADS_PER_KEY>
__global__ void
__vectorized_self_attention(
    int B, int H, int N, int D,
    float *O, int O_strideN, int O_strideD,
    float *Q, int Q_strideN, int Q_strideD,
    float *K, int K_strideN, int K_strideD,
    float *V, int V_strideN, int V_strideD,
    float *L,
    float *M
)
{
    __shared__ float Oi[Br][D_];
    __shared__ float Qi[Br][D_];
    __shared__ float Kj[Bc][D_];
    __shared__ float Vj[Bc][D_];
    __shared__ float Sij[Br][Bc];
    __shared__ float Pij[Br][Bc];
    __shared__ float Li[Br];
    __shared__ float Lij[Br];
    __shared__ float Li_new[Br];
    __shared__ float Mi[Br];
    __shared__ float Mij[Br];
    __shared__ float Mi_new[Br];
    __shared__ float PV[Br][D_];
    // get local bufs for this block's batch and head
    float *O_ = &O[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *Q_ = &Q[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *K_ = &K[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *V_ = &V[blockIdx.x * H*N*D + blockIdx.y * N*D];
    float *L_ = &L[blockIdx.x * H*N + blockIdx.y * N];
    float *M_ = &M[blockIdx.x * H*N + blockIdx.y * N];

    for(int j=0; j<N; j += Bc) {
        const int Bc_ = min(Bc, N-j);
        // load Kj to SRAM
        for(int ki=threadIdx.x; ki<Bc_; ki += blockDim.x) {
            for(int kj=threadIdx.y; kj<D; kj += blockDim.y) {
                Kj[ki][kj] = K_[(j + ki) * K_strideN + kj * K_strideD];
            }
        }
        // load Vj to SRAM
        for(int vi=threadIdx.x; vi<Bc_; vi += blockDim.x) {
            for(int vj=threadIdx.y; vj<D; vj += blockDim.y) {
                Vj[vi][vj] = V_[(j + vi) * V_strideN + vj * V_strideD];
            }
        }
        for(int i=blockIdx.z*Br; i<N; i += gridDim.z*Br) {
            const int Br_ = min(Br, N-i);
            // load Qi to SRAM
            for(int qi=threadIdx.x; qi<Br_; qi += blockDim.x) {
                for(int qj=threadIdx.y; qj<D; qj += blockDim.y) {
                    Qi[qi][qj] = Q_[(i + qi) * Q_strideN + qj * Q_strideD];
                }
            }
            // load Oi to SRAM
            for(int oi=threadIdx.x; oi<Br_; oi += blockDim.x) {
                for(int oj=threadIdx.y; oj<D; oj += blockDim.y) {
                    Oi[oi][oj] = O_[(i + oi) * O_strideN + oj * O_strideD];
                }
            }
            // load Li to SRAM
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                Li[li] = L_[i + li];
            }
            // load Mi to SRAM
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                Mi[mi] = M_[i + mi];
            }
            __syncthreads();
            // compute Sij
            for(int si=threadIdx.x; si<Br_; si += blockDim.x) {
                for(int sj=threadIdx.y; sj<Bc_; sj += blockDim.y) {
                    float dot = __warp_shfl_dot<D_, THREADS_PER_KEY>(D, (float *)&Qi[si], 1, (float *)&Kj[sj], 1);
                    // float dot = 0;
                    // for(int sk=0; sk<D; ++sk) {
                    //     dot += Qi[si][sk] * Kj[sj][sk];
                    // }
                    Sij[si][sj] = dot / sqrt(D);
                }
            }
            __syncthreads();
            // compute Mij
            for(int si=threadIdx.x; si<Br_; si += blockDim.x) {
                if(threadIdx.y == 0) {
                    float rowmax = Sij[si][0];
                    for(int sj=0; sj<Bc_; ++sj) {
                        rowmax = max(rowmax, Sij[si][sj]);
                    }
                    Mij[si] = rowmax;
                }
            }
            __syncthreads();
            // compute Pij
            for(int si=threadIdx.x; si<Br_; si += blockDim.x) {
                for(int sj=threadIdx.y; sj<Bc_; sj += blockDim.y) {
                    Pij[si][sj] = exp(Sij[si][sj] - Mij[si]);
                }
            }
            // compute Lij
            for(int pi=threadIdx.x; pi<Br_; pi += blockDim.x) {
                if(threadIdx.y == 0) {
                    float rowsum = 0;
                    for(int pj=0; pj<Bc_; ++pj) {
                        rowsum += Pij[pi][pj];
                    }
                    Lij[pi] = rowsum;
                }
            }
            // compute Mi_new
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                Mi_new[mi] = max(Mi[mi], Mij[mi]);
            }
            // compute Li_new
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                Li_new[li] = exp(Mi[li] - Mi_new[li]) * Li[li] + exp(Mij[li] - Mi_new[li]) * Lij[li];
            }
            __syncthreads();
            // compute PV
            for(int ai=threadIdx.x; ai<Br_; ai += blockDim.x) {
                for(int aj=threadIdx.y; aj<D; aj += blockDim.y) {
                    float dot = 0;
                    for(int ak=0; ak<Bc_; ++ak) {
                        dot += Pij[ai][ak] * Vj[ak][aj];
                    }
                    PV[ai][aj] = dot;
                }
            }
            // update Oi
            for(int oi=threadIdx.x; oi<Br_; oi += blockDim.x) {
                for(int oj=threadIdx.y; oj<D; oj += blockDim.y) {
                    Oi[oi][oj] = (Li[oi] * exp(Mi[oi]  - Mi_new[oi]) * Oi[oi][oj]
                                         + exp(Mij[oi] - Mi_new[oi]) * PV[oi][oj])
                                 / Li_new[oi];
                }
            }
            // update Li
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                Li[li] = Li_new[li];
            }
            // update Mi
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                Mi[mi] = Mi_new[mi];
            }
            // write O to HBM
            for(int oi=threadIdx.x; oi<Br_; oi += blockDim.x) {
                for(int oj=threadIdx.y; oj<D; oj += blockDim.y) {
                    O_[(i + oi) * O_strideN + oj * O_strideD] = Oi[oi][oj];
                }
            }
            // write L to HBM
            for(int li=threadIdx.x; li<Br_; li += blockDim.x) {
                if(threadIdx.y == 0)
                L_[i + li] = Li[li];
            }
            // write M to HBM
            for(int mi=threadIdx.x; mi<Br_; mi += blockDim.x) {
                if(threadIdx.y == 0)
                M_[i + mi] = Mi[mi];
            }
            __syncthreads();
        }       
    }
}

torch::Tensor
vectorized_self_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
)
{
    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    torch::Tensor O = torch::zeros_like(Q);
    torch::Tensor L = torch::zeros({B, H, N}, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor M = torch::zeros({B, H, N}, torch::TensorOptions().device(torch::kCUDA));

    int O_strideN = O.stride(2);
    int O_strideD = O.stride(3);
    int Q_strideN = Q.stride(2);
    int Q_strideD = Q.stride(3);
    int K_strideN = K.stride(2);
    int K_strideD = K.stride(3);
    int V_strideN = V.stride(2);
    int V_strideD = V.stride(3);

    float *O_ptr = O.data_ptr<float>();
    float *Q_ptr = Q.data_ptr<float>();
    float *K_ptr = K.data_ptr<float>();
    float *V_ptr = V.data_ptr<float>();
    float *L_ptr = L.data_ptr<float>();
    float *M_ptr = M.data_ptr<float>();

    dim3 grid(
        B,
        H,
        (N + Br - 1) / Br
    );
    dim3 block(Br, D, 1);
    __vectorized_self_attention<1><<<grid, block>>>(
        B, H, N, D,
        O_ptr, O_strideN, O_strideD,
        Q_ptr, Q_strideN, Q_strideD,
        K_ptr, K_strideN, K_strideD,
        V_ptr, V_strideN, V_strideD,
        L_ptr,
        M_ptr
    );
    return O;
}
