
#include <stdio.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 16777216

typedef struct timeval tval;

/**
 * Helper method to generate a very naive "hash".
 */
float generate_hash(int n, float *y)
{
    float hash = 0.0f;
    
    for (int i = 0; i < n; i++)
    {
        hash += y[i];
    }
    
    return hash;
}

/**
 * Helper method that calculates the elapsed time between two time intervals (in milliseconds).
 */
long get_elapsed(tval t0, tval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;
}

/**
 * SAXPY reference implementation using the CPU.
 */
void cpu_saxpy(int n, float a, float *x, float *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = a * x[i] + y[i];
    }
}

/**
 * SAXPY implementation using the GPU.
 */
__global__ void gpu_saxpy(int n, float a, float *d_x, float *d_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n)
    {
        d_y[i] = a * d_x[i] + d_y[i];
    }
}

int main(int argc, char **argv)
{
    float a     = 0.0f;
    float *x    = NULL;
    float *y    = NULL;
    float *d_x  = NULL;
    float *d_y  = NULL;
    float error = 0.0f;
    dim3  grid((ARRAY_SIZE + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
    dim3  block(BLOCK_SIZE);
    
    // Make sure the constant is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The constant is missing!\n");
        return -1;
    }
    
    // Retrieve the constant and allocate the arrays on the CPU
    a = atof(argv[1]);
    x = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    y = (float *)malloc(sizeof(float) * ARRAY_SIZE);
    
    // Initialize them with fixed values
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        x[i] = 0.1f;
        y[i] = 0.2f;
    }
    
    // Allocate the arrays on the GPU and copy the content
    cudaMalloc(&d_x,   ARRAY_SIZE * sizeof(float));
    cudaMalloc(&d_y,   ARRAY_SIZE * sizeof(float));
    cudaMemcpy(d_x, x, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Call the CPU code
    cpu_saxpy(ARRAY_SIZE, a, x, y);
    
    // Calculate the "hash" of the result from the CPU
    error = generate_hash(ARRAY_SIZE, y);
    
    // Call the GPU code and copy back the result
    gpu_saxpy<<<grid, block>>>(ARRAY_SIZE, a, d_x, d_y);
    
    cudaMemcpy(y, d_y, ARRAY_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate the "hash" of the result from the GPU
    error = fabsf(error - generate_hash(ARRAY_SIZE, y));
    
    // Confirm that the execution has finished
    printf("Execution finished (error=%.6f).\n", error);
    
    if (error > 0.0001f)
    {
        fprintf(stderr, "Error: The solution is incorrect!\n");
    }
    
    // Release all the allocations
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    
    return 0;
}

