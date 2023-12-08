#include "util.cuh"
#include "math.cuh"

__global__ void
__dot0(
    int D,
    float *dot,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    float result = __simple_block_dot(D, A, A_strideD, B, B_strideD);
    if(threadIdx.x == 0)
        *dot = result;
}

template <int D_>
__global__ void
__dot1(
    int D,
    float *dot,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    float result = __shared_mem_block_dot<D_>(D, A, A_strideD, B, B_strideD);
    if (threadIdx.x == 0)
        *dot = result;
}

template <int D_>
__global__ void
__dot2(
    int D,
    float *dot,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    float result = __warp_shfl_block_dot<D_>(D, A, A_strideD, B, B_strideD);
    if (threadIdx.x == 0)
        *dot = result;
}

template <int D_>
__global__ void
__dot3(
    int D,
    float *dot,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    float result = __single_sync_block_dot<D_>(D, A, A_strideD, B, B_strideD);
    if (threadIdx.x == 0)
        *dot = result;
}

template <int D_, int THREADS_PER_KEY>
__global__ void
__dot4(
    int D,
    float *dot,
    float *A, int A_strideD,
    float *B, int B_strideD
)
{
    static const int ELEMS_PER_THREAD = D_ / THREADS_PER_KEY;
    A = &A[threadIdx.x * ELEMS_PER_THREAD];
    B = &B[threadIdx.x * ELEMS_PER_THREAD];
    float result = __warp_shfl_dot<D_, THREADS_PER_KEY>(D, A, A_strideD, B, B_strideD);
    if (threadIdx.x == 0)
        *dot = result;
}

__global__ void
__init(
    int D,
    float *A,
    int A_strideD
)
{
    for(int i = blockIdx.x*blockDim.x+threadIdx.x; i<D; i += gridDim.x*blockDim.x)
        A[i * A_strideD] = i % 10;
}

__global__ void
__print(
    int D,
    float *A,
    int A_strideD
)
{
    if(threadIdx.x == 0)
        __print_mat(D, A_strideD, A);
}

int main()
{
    static const int D = 1024; // must be a power of 2
    float *A;
    float *B;
    float *dot0;
    float *dot1;
    float *dot2;
    float *dot3;
    float *dot4;
    cudaMalloc(&A, D * sizeof(float));
    cudaMalloc(&B, D * sizeof(float));
    cudaMalloc(&dot0, 1 * sizeof(float));
    cudaMalloc(&dot1, 1 * sizeof(float));
    cudaMalloc(&dot2, 1 * sizeof(float));
    cudaMalloc(&dot3, 1 * sizeof(float));
    cudaMalloc(&dot4, 1 * sizeof(float));
    __init<<<1, D>>>(D, A, 1);
    __init<<<1, D>>>(D, B, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int N_trials = 5;
    float dot0_ms = 0;
    float dot1_ms = 0;
    float dot2_ms = 0;
    float dot3_ms = 0;
    float dot4_ms = 0;
    for(int i=0; i<N_trials; ++i) {
        float ms;
        cudaEventRecord(start);
        __dot0   <<<1, D>>>(D, dot0, A, 1, B, 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        dot0_ms += ms;
        cudaEventRecord(start);
        __dot1<D><<<1, D>>>(D, dot1, A, 1, B, 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        dot1_ms += ms;
        cudaEventRecord(start);
        __dot2<D><<<1, D>>>(D, dot2, A, 1, B, 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        dot2_ms += ms;
        cudaEventRecord(start);
        __dot3<D><<<1, D>>>(D, dot3, A, 1, B, 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        dot3_ms += ms;
        static const int THREADS_PER_KEY = 8;
        cudaEventRecord(start);
        __dot4<D, THREADS_PER_KEY><<<1, THREADS_PER_KEY>>>(D, dot4, A, 1, B, 1);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        dot4_ms += ms;
    }
    dot0_ms /= N_trials;
    dot1_ms /= N_trials;
    dot2_ms /= N_trials;
    dot3_ms /= N_trials;
    dot4_ms /= N_trials;

    __print<<<1, D>>>(D, A, 1);
    __print<<<1, D>>>(D, B, 1);
    __print<<<1, 1>>>(1, dot0, 1);
    __print<<<1, 1>>>(1, dot1, 1);
    __print<<<1, 1>>>(1, dot2, 1);
    __print<<<1, 1>>>(1, dot3, 1);
    __print<<<1, 1>>>(1, dot4, 1);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    printf("dot0: %f\n", dot0_ms);
    printf("dot1: %f\n", dot1_ms);
    printf("dot2: %f\n", dot2_ms);
    printf("dot3: %f\n", dot3_ms);
    printf("dot4: %f\n", dot4_ms);
    return 0;
}
