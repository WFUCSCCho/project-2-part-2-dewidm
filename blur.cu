#include <iostream>
#include <vector>
#include <cuda.h>
#include <vector_types.h>

#define BLUR_SIZE 16 
// TILE_SIZE matches the block dimension
#define TILE_SIZE 16
// The shared memory needs to be larger than the tile to account for the "halo" pixels
#define SHMEM_SIZE (TILE_SIZE + 2 * BLUR_SIZE)

#include "bitmap_image.hpp"

using namespace std;

// Improved Kernel using Shared Memory
__global__ void blurKernelShared(uchar3 *in, uchar3 *out, int width, int height) {
    // Shared memory for the tile plus the halo (boundary) pixels needed for blurring
    __shared__ uchar3 s_pixels[SHMEM_SIZE][SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    // Collaborative load: Threads work together to fill shared memory including the halo
    // We iterate to ensure all SHMEM_SIZE x SHMEM_SIZE pixels are loaded
    for (int i = ty; i < SHMEM_SIZE; i += TILE_SIZE) {
        for (int j = tx; j < SHMEM_SIZE; j += TILE_SIZE) {
            int curRow = (blockIdx.y * TILE_SIZE - BLUR_SIZE) + i;
            int curCol = (blockIdx.x * TILE_SIZE - BLUR_SIZE) + j;

            if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                s_pixels[i][j] = in[curRow * width + curCol];
            } else {
                s_pixels[i][j] = make_uchar3(0, 0, 0); // Black for out of bounds
            }
        }
    }

    __syncthreads(); // Ensure all threads finished loading to shmem

    if (col < width && row < height) {
        int3 pixVal = {0, 0, 0};
        int pixels = 0;

        // Computation now pulls from fast shared memory s_pixels
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++) {
                // Shift indices to match shared memory layout (center is at BLUR_SIZE)
                uchar3 p = s_pixels[ty + BLUR_SIZE + blurRow][tx + BLUR_SIZE + blurCol];
                
                // Only count if it wasn't a padded/out-of-bounds pixel from global load
                // (Simplified: for a standard box blur, you'd check original global bounds)
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    pixVal.x += p.x;
                    pixVal.y += p.y;
                    pixVal.z += p.z;
                    pixels++;
                }
            }
        }

        out[row * width + col].x = (unsigned char)(pixVal.x / pixels);
        out[row * width + col].y = (unsigned char)(pixVal.y / pixels);
        out[row * width + col].z = (unsigned char)(pixVal.z / pixels);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "format: " << argv[0] << " { filename.bmp }" << endl;
        exit(1);
    }
    
    bitmap_image bmp(argv[1]);
    if(!bmp) { cerr << "Image not found" << endl; exit(1); }

    int height = bmp.height();
    int width = bmp.width();
    
    vector<uchar3> input_image;
    rgb_t color;
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            bmp.get_pixel(x, y, color);
            input_image.push_back( {color.red, color.green, color.blue} );
        }
    }

    vector<uchar3> output_image(input_image.size());
    uchar3 *d_in, *d_out;
    size_t img_size = input_image.size() * sizeof(uchar3);
    
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMemcpy(d_in, input_image.data(), img_size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // CUDA Events for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up call (GPU initialization hint)
    blurKernelShared<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);

    // Actual timed run
    cudaEventRecord(start);
    blurKernelShared<<<dimGrid, dimBlock>>>(d_in, d_out, width, height);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time to blur: " << milliseconds << " ms" << endl;

    cudaMemcpy(output_image.data(), d_out, img_size, cudaMemcpyDeviceToHost);
    
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int pos = y * width + x;
            bmp.set_pixel(x, y, output_image[pos].x, output_image[pos].y, output_image[pos].z);
        }
    }

    bmp.save_image("./blurred.bmp");
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}