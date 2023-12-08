#ifndef MATH_H
#define MATH_H

__device__ float
__simple_block_dot(
    int D,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    float dot = 0;
    for(int i=0; i<D; ++i) {
        dot += A[i * A_strideD] * B[i * B_strideD];
    }
    return dot;
}

template <int D_>
__device__ float
__shared_mem_block_dot(
    int D,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    __shared__ float tmp[D_];
    if(threadIdx.x < D)
        tmp[threadIdx.x] = A[threadIdx.x * A_strideD] * B[threadIdx.x * B_strideD];

    for(int offset=D>>1; offset>=1; offset>>=1) {
        __syncthreads();
        if(threadIdx.x < offset)
            tmp[threadIdx.x] += tmp[threadIdx.x + offset];
    }
    return tmp[0];
}

template <int D_>
__device__ float
__warp_shfl_block_dot(
    int D,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    __shared__ float tmp[D_];
    if(threadIdx.x < D)
        tmp[threadIdx.x] = A[threadIdx.x * A_strideD] * B[threadIdx.x * B_strideD];

    for(int s=D>>1; s>=32; s>>=1) {
        __syncthreads();
        if(threadIdx.x < s)
            tmp[threadIdx.x] += tmp[threadIdx.x + s];
    }
    __syncthreads();

    unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < D);
    float val = 0;
    if(threadIdx.x < D && threadIdx.x < 32) {
        val = tmp[threadIdx.x];
        for(int offset=16; offset>=1; offset>>=1)
            val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

template <int D_>
__device__ float
__single_sync_block_dot(
    int D,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    __shared__ float tmp[32];

    unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < D);
    float val = 0;
    if(threadIdx.x < D) {
        val = A[threadIdx.x * A_strideD] * B[threadIdx.x * B_strideD];
        for(int offset=16; offset>=1; offset>>=1)
            val += __shfl_down_sync(mask, val, offset);
        if(threadIdx.x % 32 == 0)
            tmp[threadIdx.x / 32] = val;
    }
    __syncthreads();

    mask = __ballot_sync(0xffffffff, threadIdx.x < D / 32);
    if(threadIdx.x < D / 32 && threadIdx.x < 32) {
        val = tmp[threadIdx.x];
        for(int offset=16; offset>=1; offset>>=1)
            val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

// this device function assumes A and B already have the needed
// offset per thread
template <int D_, int THREADS_PER_KEY>
__device__ float
__warp_shfl_dot(
    int D,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    static const int ELEMS_PER_THREAD = D_ / THREADS_PER_KEY;
    float dot = A[0] * B[0];
    for(int i=1; i<ELEMS_PER_THREAD; ++i)
        dot += A[i * A_strideD] * B[i * B_strideD];
    for(int mask=THREADS_PER_KEY>>1; mask>=1; mask>>=1)
        dot += __shfl_xor_sync(0xffffffff, dot, mask);
    return dot;
}

#endif
