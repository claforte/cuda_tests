#include "../common/utils.h"

// Matrix multiplication kernel using one thread per one P element
__global__ void matMulKernel(const float* M, const float* N, float* P, int width)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < width && col < width)
    {
        float Pval = 0;
        for (int k=0; k<width; ++k)
        {
            Pval += M[row*width + k] * N[k*width + col];
        }
        P[row*width + col] = Pval;
    }
}

// Test P=MxN where all 3 are matrice.
int main(void)
{
    int width = 64, height = 64;
    int numElements = width * height;
    size_t size = numElements * sizeof(float);
    printf("Multiplication of two (%d x %d) matrice\n", height, width);

    float* M_h = (float*) malloc(size);
    float* N_h = (float*) malloc(size);
    float* P_h = (float*) malloc(size);

    if (M_h == NULL || N_h == NULL || P_h == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrice!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host matrice
    for (int r=0; r<width; ++r)
        for (int c=0; c<width; c++)
        {
            M_h[r*width+c] = rand() / (float)RAND_MAX;
            N_h[r*width+c] = rand() / (float)RAND_MAX;
        }
    
    // Allocate the device vectors
    float* M_d = NULL;
    float* N_d = NULL;
    float* P_d = NULL;

    CUDA_CHECK(cudaMalloc((float**) &M_d, size));
    CUDA_CHECK(cudaMalloc((float**) &N_d, size));
    CUDA_CHECK(cudaMalloc((float**) &P_d, size));

    // Copy vectors from host memory to device memory
    CUDA_CHECK(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));
    
    // Launch the kernel
    dim3 threadsPerBlock(16, 16);  // 256 threads total
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (width + threadsPerBlock.y - 1) / threadsPerBlock.y);
    printf("CUDA kernel launch with (%d x %d) blocks of (%d x %d) threads\n", blocksPerGrid.y, blocksPerGrid.x, threadsPerBlock.y, threadsPerBlock.x);

    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(M_d, N_d, P_d, width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result matrix from device memory
    CUDA_CHECK(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));

    // Verify that the result vector is correct
    for (int r=0; r<width; ++r)
        for (int c=0; c<width; c++)
        {
            float Pval = 0;
            for (int k=0; k<width; ++k)
                Pval += M_h[r*width + k] * N_h[k*width + c];
            
            if (fabs(Pval - P_h[r*width+c]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at row %d, column %d!\n", r, c);
                exit(EXIT_FAILURE);
            }
        }
    printf("Test passed!\n");

    // Free device global memory
    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    // Free host memory
    free(M_h);
    free(N_h);
    free(P_h);
    
    printf("Done\n");
    return 0;
}