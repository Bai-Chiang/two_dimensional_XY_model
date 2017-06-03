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
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(
            seed, // seed value is same for each core
            index, // sequence number for each core is it's index
            0, // offset is 0
            &states [index]
            );
}

__global__ void initialize (float* spins, int* length, curandState_t* states) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    //spins[index] = curand_uniform() * 2 * M_PI;
    //spins[index] = float(M_PI);
    for (int i = 0; i < 10000; ++i) {
        spins[index] = curand_uniform(&states[index]) * 2*M_PI;
    }
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
    long long n_sample = 100;
    long long warm_up_steps = 100;
    int length = 200;
    unsigned int seed = 0;

    int n_block = length;
    int n_threads_per_block = length;
    size_t size = length * length;

    // --------------------------------------------------------------------------------------------
    curandState_t* states; // used to store random state for each core
    cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
    init_rand<<<n_block, n_threads_per_block>>>(seed, states); // initialize for all states
    
    //std::cout << "init_rand completed" << std::endl;
    
    // allocate memory in device
    float* d_spins; // device copy of a spin system (pp_spins[i])
    int* d_length; // device copy of length
    cudaMalloc((void**)&d_spins, size * sizeof(float));
    cudaMalloc((void**)&d_length, sizeof(int));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);

    /* -----------------------------------------------------------------
     * get samples and initialize them
     */

    float** pp_spins = new float* [n_sample];
    for (long long i = 0; i < n_sample; ++i) {
        pp_spins[i] = new float [size];
    }

    for (long long i = 0; i < n_sample; ++i) {
        // copy memory from host to device
        cudaMemcpy(d_spins, pp_spins[i], size * sizeof(float), cudaMemcpyHostToDevice);

        initialize<<<n_block, n_threads_per_block>>>(d_spins, d_length, states);

        //copy memory from device to host
        cudaMemcpy(pp_spins[i], d_spins, size * sizeof(float), cudaMemcpyDeviceToHost);
        
    }

    /*
    // -------------------------------------
    for (long long i = 0; i < n_sample; ++i) {
        print_spins(pp_spins[i], length, 5, 10);
    }
    */

    // -----------------------------------------------------------------
    cudaFree(&d_length); cudaFree(d_spins);
    cudaFree(states); 
    // free pp_spins
    for (long long i = 0; i < n_sample; ++i) {
        delete[] pp_spins[i];
    }
    delete[] pp_spins;
}
