
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define BLOCK_SIZE  16
#define HEADER_SIZE 122

typedef unsigned char BYTE;

/**
 * Structure that represents a BMP image.
 */
typedef struct
{
    int   width;
    int   height;
    float *data;
} BMPImage;

typedef struct timeval tval;

BYTE g_info[HEADER_SIZE]; // Reference header

/**
 * Reads a BMP 24bpp file and returns a BMPImage structure.
 * Thanks to https://stackoverflow.com/a/9296467
 */
BMPImage readBMP(char *filename)
{
    BMPImage bitmap = { 0 };
    int      size   = 0;
    BYTE     *data  = NULL;
    FILE     *file  = fopen(filename, "rb");
    
    // Read the header (expected BGR - 24bpp)
    fread(g_info, sizeof(BYTE), HEADER_SIZE, file);

    // Get the image width / height from the header
    bitmap.width  = *((int *)&g_info[18]);
    bitmap.height = *((int *)&g_info[22]);
    size          = *((int *)&g_info[34]);
    
    // Read the image data
    data = (BYTE *)malloc(sizeof(BYTE) * size);
    fread(data, sizeof(BYTE), size, file);
    
    // Convert the pixel values to float
    bitmap.data = (float *)malloc(sizeof(float) * size);
    
    for (int i = 0; i < size; i++)
    {
        bitmap.data[i] = (float)data[i];
    }
    
    fclose(file);
    free(data);
    
    return bitmap;
}

/**
 * Writes a BMP file in grayscale given its image data and a filename.
 */
void writeBMPGrayscale(int width, int height, float *image, char *filename)
{
    FILE *file = NULL;
    
    file = fopen(filename, "wb");
    
    // Write the reference header
    fwrite(g_info, sizeof(BYTE), HEADER_SIZE, file);
    
    // Unwrap the 8-bit grayscale into a 24bpp (for simplicity)
    for (int h = 0; h < height; h++)
    {
        int offset = h * width;
        
        for (int w = 0; w < width; w++)
        {
            BYTE pixel = (BYTE)((image[offset + w] > 255.0f) ? 255.0f :
                                (image[offset + w] < 0.0f)   ? 0.0f   :
                                                               image[offset + w]);
            
            // Repeat the same pixel value for BGR
            fputc(pixel, file);
            fputc(pixel, file);
            fputc(pixel, file);
        }
    }
    
    fclose(file);
}

/**
 * Releases a given BMPImage.
 */
void freeBMP(BMPImage bitmap)
{
    free(bitmap.data);
}

/**
 * Checks if there has been any CUDA error. The method will automatically print
 * some information and exit the program when an error is found.
 */
void checkCUDAError()
{
    cudaError_t cudaError = cudaGetLastError();
    
    if(cudaError != cudaSuccess)
    {
        printf("CUDA Error: Returned %d: %s\n", cudaError,
                                                cudaGetErrorString(cudaError));
        exit(-1);
    }
}

/**
 * Calculates the elapsed time between two time intervals (in milliseconds).
 */
long get_elapsed(tval t0, tval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000;
}

/**
 * Stores the result image and prints a message.
 */
void store_result(int index, long elapsed_cpu, long elapsed_gpu,
                     int width, int height, float *image)
{
    char path[255];
    
    sprintf(path, "images/lab02_result_%d.bmp", index);
    writeBMPGrayscale(width, height, image, path);
    
    printf("Step #%d Completed - Result stored in \"%s\".\n", index, path);
    printf("Elapsed CPU: %ldms / ", elapsed_cpu);
    
    if (elapsed_gpu == 0)
    {
        printf("[GPU version not available]\n");
    }
    else
    {
        printf("Elapsed GPU: %ldms\n", elapsed_gpu);
    }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the CPU.
 */
void cpu_grayscale(int width, int height, float *image, float *image_out)
{
    for (int h = 0; h < height; h++)
    {
        int offset_out = h * width;      // 1 color per pixel
        int offset     = offset_out * 3; // 3 colors per pixel
        
        for (int w = 0; w < width; w++)
        {
            float *pixel = &image[offset + w * 3];
            
            // Convert to grayscale following the "luminance" model
            image_out[offset_out + w] = pixel[0] * 0.0722f + // B
                                        pixel[1] * 0.7152f + // G
                                        pixel[2] * 0.2126f;  // R
        }
    }
}

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__ void gpu_grayscale(int width, int height, float *image, float *image_out)
{
    ////////////////
    // TO-DO #4.2 /////////////////////////////////////////////
    // Implement the GPU version of the grayscale conversion //
    ///////////////////////////////////////////////////////////
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the CPU.
 */
float cpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    float pixel = 0.0f;
    
    for (int h = 0; h < filter_dim; h++)
    {
        int offset        = h * stride;
        int offset_kernel = h * filter_dim;
        
        for (int w = 0; w < filter_dim; w++)
        {
            pixel += image[offset + w] * matrix[offset_kernel + w];
        }
    }
    
    return pixel;
}

/**
 * Applies a 3x3 convolution matrix to a pixel using the GPU.
 */
__device__ float gpu_applyFilter(float *image, int stride, float *matrix, int filter_dim)
{
    ////////////////
    // TO-DO #5.2 ////////////////////////////////////////////////
    // Implement the GPU version of cpu_applyFilter()           //
    //                                                          //
    // Does it make sense to have a separate gpu_applyFilter()? //
    //////////////////////////////////////////////////////////////
    
    return 0.0f;
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the CPU.
 */
void cpu_gaussian(int width, int height, float *image, float *image_out)
{
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    for (int h = 0; h < (height - 2); h++)
    {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++)
        {
            image_out[offset + (w + 1)] = cpu_applyFilter(&image[offset_t + w],
                                                          width, gaussian, 3);
        }
    }
}

/**
 * Applies a Gaussian 3x3 filter to a given image using the GPU.
 */
__global__ void gpu_gaussian(int width, int height, float *image, float *image_out)
{
    float gaussian[9] = { 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f,
                          2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f,
                          1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f };
    
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (index_x < (width - 2) && index_y < (height - 2))
    {
        int offset_t = index_y * width + index_x;
        int offset   = (index_y + 1) * width + (index_x + 1);
        
        image_out[offset] = gpu_applyFilter(&image[offset_t],
                                            width, gaussian, 3);
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the CPU.
 */
void cpu_sobel(int width, int height, float *image, float *image_out)
{
    float sobel_x[9] = { 1.0f,  0.0f, -1.0f,
                         2.0f,  0.0f, -2.0f,
                         1.0f,  0.0f, -1.0f };
    float sobel_y[9] = { 1.0f,  2.0f,  1.0f,
                         0.0f,  0.0f,  0.0f,
                        -1.0f, -2.0f, -1.0f };
    
    for (int h = 0; h < (height - 2); h++)
    {
        int offset_t = h * width;
        int offset   = (h + 1) * width;
        
        for (int w = 0; w < (width - 2); w++)
        {
            float gx = cpu_applyFilter(&image[offset_t + w], width, sobel_x, 3);
            float gy = cpu_applyFilter(&image[offset_t + w], width, sobel_y, 3);
            
            // Note: The output can be negative or exceed the max. color value
            // of 255. We compensate this afterwards while storing the file.
            image_out[offset + (w + 1)] = sqrtf(gx * gx + gy * gy);
        }
    }
}

/**
 * Calculates the gradient of an image using a Sobel filter on the GPU.
 */
__global__ void gpu_sobel(int width, int height, float *image, float *image_out)
{
    ////////////////
    // TO-DO #6.1 /////////////////////////////////////
    // Implement the GPU version of the Sobel filter //
    ///////////////////////////////////////////////////
}

int main(int argc, char **argv)
{
    BMPImage bitmap          = { 0 };
    float    *d_bitmap       = { 0 };
    float    *image_out[2]   = { 0 };
    float    *d_image_out[2] = { 0 };
    int      image_size      = 0;
    tval     t[2]            = { 0 };
    long     elapsed[2]      = { 0 };
    dim3     grid(1);                       // The grid will be defined later
    dim3     block(BLOCK_SIZE, BLOCK_SIZE); // The block size will not change
    
    // Make sure the filename is provided
    if (argc != 2)
    {
        fprintf(stderr, "Error: The filename is missing!\n");
        return -1;
    }
    
    // Read the input image and update the grid dimension
    bitmap     = readBMP(argv[1]);
    image_size = bitmap.width * bitmap.height;
    grid       = dim3(((bitmap.width  + (BLOCK_SIZE - 1)) / BLOCK_SIZE),
                      ((bitmap.height + (BLOCK_SIZE - 1)) / BLOCK_SIZE));
    
    printf("Image opened (width=%d height=%d).\n", bitmap.width, bitmap.height);
    
    // Allocate the intermediate image buffers for each step
    for (int i = 0; i < 2; i++)
    {
        image_out[i] = (float *)calloc(image_size, sizeof(float));
        
        cudaMalloc(&d_image_out[i], image_size * sizeof(float));
        cudaMemset(d_image_out[i], 0, image_size * sizeof(float));
    }

    cudaMalloc(&d_bitmap, image_size * sizeof(float) * 3);
    cudaMemcpy(d_bitmap, bitmap.data,
               image_size * sizeof(float) * 3, cudaMemcpyHostToDevice);
    
    // Step 1: Convert to grayscale
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_grayscale(bitmap.width, bitmap.height, bitmap.data, image_out[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        // gpu_grayscale<<<grid, block>>>(bitmap.width, bitmap.height,
        //                                d_bitmap, d_image_out[0]);
        
        // cudaMemcpy(image_out[0], d_image_out[0],
        //            image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image in grayscale
        store_result(1, elapsed[0], elapsed[1], bitmap.width, bitmap.height, image_out[0]);
    }
    
    // Step 2: Apply a 3x3 Gaussian filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_gaussian(bitmap.width, bitmap.height, image_out[0], image_out[1]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        // gpu_gaussian<<<grid, block>>>(bitmap.width, bitmap.height,
        //                               d_image_out[0], d_image_out[1]);
        
        // cudaMemcpy(image_out[1], d_image_out[1],
        //            image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the result image with the Gaussian filter applied
        store_result(2, elapsed[0], elapsed[1], bitmap.width, bitmap.height, image_out[1]);
    }
    
    // Step 3: Apply a Sobel filter
    {
        // Launch the CPU version
        gettimeofday(&t[0], NULL);
        cpu_sobel(bitmap.width, bitmap.height, image_out[1], image_out[0]);
        gettimeofday(&t[1], NULL);
        
        elapsed[0] = get_elapsed(t[0], t[1]);
        
        // Launch the GPU version
        gettimeofday(&t[0], NULL);
        // gpu_sobel<<<grid, block>>>(bitmap.width, bitmap.height,
        //                            d_image_out[1], d_image_out[0]);
        
        // cudaMemcpy(image_out[0], d_image_out[0],
        //            image_size * sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t[1], NULL);
        
        elapsed[1] = get_elapsed(t[0], t[1]);
        
        // Store the final result image with the Sobel filter applied
        store_result(3, elapsed[0], elapsed[1], bitmap.width, bitmap.height, image_out[0]);
    }
    
    // Release the allocated memory
    for (int i = 0; i < 2; i++)
    {
        free(image_out[i]);
        cudaFree(d_image_out[i]);
    }
    
    freeBMP(bitmap);
    cudaFree(d_bitmap);
    
    return 0;
}

