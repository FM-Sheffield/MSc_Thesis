#pragma once
#ifdef __CUDACC__

/*
 *
 *      From CUDA By Example An Introduction to General-Purpose GPU Programming”
 *  by Jason Sanders and Edward Kandrot, Addison-Wesley, Upper Saddle River, NJ, 2011
 *
 */
// Macro for handle errors
#include<stdio.h>
__host__ static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



/* Function to check for CUDA runtime errors */
static void checkCUDAError(const char* msg) {
	/* Get last error */
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
    	/* Print the error */
        printf("Cuda error: %s %s\n",msg, cudaGetErrorString( err));
        /* Abort the program */
        exit(EXIT_FAILURE);
    }
}
#endif
