#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors.
// Uses do/while(0), a common C idiom for macros to ensure the macro works
// correctly in all contexts, especially in if-else statements.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0) 
