//
// Created by Baiqiang Qiang on 30/05/2017.
//

#include <cstdio>
#include <iostream>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

/* ------------------------------------------------------------------------------------------------
 * initial seed and sequence for all device cores
 * all cores will have same seed, but different sequence number
 * so each core will generate different random number series 
 */
__global__ void init_rand(unsigned int seed, curandState_t* states) {
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    curand_init(
            seed, // seed value is same for each core
            index, // sequence number for each core is it's index
            0, // offset is 0
            &states [index]
            );
}

/* ------------------------------------------------------------------------------------------------
 * initialize the spins with random direction from [0, 2*pi]
 */
__global__ void initialize (float* d_spins, curandState_t* states) {
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    d_spins[index] = curand_uniform(&states[index]) * 2*M_PI;
}

void print_spins (float* spins, int length, int print_dim1, int print_dim2) {
    for (int i = 0; i < print_dim1; ++i) {
        for (int j = 0; j < print_dim2; ++j) {
            printf("%f ", spins[i*length + j]);
            //std::cout << spins[i*length + j] << " ";
        }
        //std::cout << std::endl;
        printf("\n");
    }
    //std::cout << std::endl;
    printf("\n");
}

// ================================================================================================

int main() {
    long long n_sample = 10;
    long long warm_up_steps = 100;
    int length = 512; // 2^n, n >= 5 
    unsigned int seed = 0;
    long long size = length * length;
    int threads_per_block = 1024;
    int blocks = size/threads_per_block;

    // --------------------------------------------------------------------------------------------
    curandState_t* states; // used to store random state for each core
    cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
    init_rand<<<blocks, threads_per_block>>>(seed, states); // initialize for all states
    
    //std::cout << "init_rand completed" << std::endl;
    
    // allocate memory in device
    float* d_spins; // device copy of a spin system (pp_spins[i])
    int* d_length; // device copy of length
    cudaMalloc((void**)&d_spins, size * sizeof(float));
    cudaMalloc((void**)&d_length, sizeof(int));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);

    float* p_spins = new float[size];
    cudaDeviceSynchronize();

    
    for (long long i = 0; i < n_sample; ++i) {
       // initialize spins
        initialize<<<blocks, threads_per_block>>>(d_spins, states);

        // copy memory from device to host
        cudaMemcpy(p_spins, d_spins, size * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << i << std::endl;
        cudaDeviceSynchronize();

    }
    
    print_spins (p_spins, length, 10, 15);

    // -----------------------------------------------------------------
    cudaFree(&d_length); cudaFree(d_spins);
    cudaFree(states); 
    delete[] p_spins;
    return 0;
}
