#include <buffer_manager.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cuda.h>

#include <stdio.h>

__host__ unsigned char *blank_buffer(int width, int height){
    unsigned char *f;
    cudaMalloc(&f, width * height * 3);
    return f;
}

__global__ void fade_buffer_kernel(unsigned char* buff, int width, int height){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    printf("X: %d  | Y: %d\n", x, y);

    // std::cout << "X: " << x << "Y: " << y << std::endl;

    if(y > height || x > width) return;

    int pixel_index = y * width * 3 + x * 3;

    buff[pixel_index] = (unsigned char)(255.99f * (float(x) / float(width)));
    buff[pixel_index + 1] = (unsigned char)(255.99f * (float(y) / float(height)));
    buff[pixel_index + 2] = 60;

}

void save_ppm(const char *path, unsigned char *buff, int width, int height){
    std::ofstream fp;
    fp.open(path);

    fp << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height-1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {

            // std::cout << "X: " << i << "Y: " << j << std::endl;
            // printf("X: %d  | Y: %d\n", i, j);
            int pixel_index = j * 3 * width + i * 3;

            unsigned char r = buff[pixel_index];
            unsigned char g = buff[pixel_index + 1];
            unsigned char b = buff[pixel_index + 2];
   
            fp << int(r) << " " << int(g) << " " << int(b) << "\n";
        }
    }

    fp.close();
}

int main(int argc, char const *argv[])
{

    const int width = 800, height = 600;
    const int num_thread_per_block = 128, num_blocks = width * height / num_thread_per_block ;

    auto buff = blank_buffer(800, 600);

    fade_buffer_kernel<<<num_blocks, num_thread_per_block>>>(buff, width, height);

    unsigned char *hostbuff = (unsigned char *)malloc(width * height * 3);
    cudaMemcpy(hostbuff, buff, width * height * 3, cudaMemcpyDeviceToHost);
    save_ppm("testimage.ppm", hostbuff, 800, 600);

    cudaFree(buff);
    free(hostbuff);
    return 0;
}
