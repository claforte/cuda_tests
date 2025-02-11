#include "../common/utils.h"

// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("Vector addition of %d elements\n", numElements);

    float* A_h = (float*) malloc(size);
    float* B_h = (float*) malloc(size);
    float* C_h = (float*) malloc(size);

    if (A_h == NULL || B_h == NULL || C_h == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host vectors
    for (int i=0; i<numElements; ++i)
    {
        A_h[i] = rand() / (float)RAND_MAX;
        B_h[i] = rand() / (float)RAND_MAX;
    }

    // Allocate the device vectors
    float* A_d = NULL;
    float* B_d = NULL;
    float* C_d = NULL;

    CUDA_CHECK(cudaMalloc((float**) &A_d, size));
    CUDA_CHECK(cudaMalloc((float**) &B_d, size));
    CUDA_CHECK(cudaMalloc((float**) &C_d, size));

    // Copy vectors from host memory to device memory
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
    
    // Launch the vectorAdd kernel
    int threadsPerBlock = 256;
    // TODO: try ceil instead.
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, numElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result vector from device memory
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    // Verify that the result vector is correct
    for (int i=0; i<numElements; ++i)
    {
        if (fabs(A_h[i] + B_h[i] - C_h[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test passed!\n");

    // Free device global memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    // Free host memory
    free(A_h);
    free(B_h);
    free(C_h);
    
    printf("Done\n");
    return 0;
}