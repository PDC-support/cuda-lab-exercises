
#include <stdio.h>

/**
 * CPU version of our CUDA Hello World!
 */
void cpu_helloworld()
{
    printf("Hello from the CPU!\n");
}

/**
 * GPU version of our CUDA Hello World!
 */
__global__ void gpu_helloworld()
{
    int threadId = threadIdx.x;
    printf("Hello from the GPU! My threadId is %d\n", threadId);
}

int main(int argc, char **argv)
{
    dim3 grid(1);     // 1 block in the grid
    dim3 block(32);   // 32 threads per block
    
    // Call the CPU version
    cpu_helloworld();
    
    // Call the GPU version
    gpu_helloworld<<<grid, block>>>();
    
    ////////////////
    // TO-DO #1.2 ////////////////////
    // Introduce your changes here! //
    //////////////////////////////////
    
    return 0;
}

