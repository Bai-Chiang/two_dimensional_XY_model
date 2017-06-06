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
__global__ 
void init_rand(unsigned int seed, curandState_t* states) {
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
__global__ 
void initialize (float* spins, curandState_t* states) {
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    //spins[index] = curand_uniform(&states[index]) * 2*M_PI;
    spins[index] = index;
}


/* ------------------------------------------------------------------------------------------------
 * warm up the system
 * n_itter is the itteration times for each thread
 * the total warm_up_step for the whole system is n_itter * n_block * n_threads_per_block
 */
__global__ 
void warm_up (float* spins, int* p_length, long long* n_itter) {
    int dim1 = 9;
    int dim2 = 9;
    float upper_spin = 0;
    int length = *p_length;
    
    // get upper spin
    if (dim1 == 0) {
        upper_spin = spins[(length-1) * length + dim2];
    } else {
        upper_spin = spins[(dim1-1) * length + dim2];
    }
    spins[0] = upper_spin;
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
    long long n_sample = 1;
    int length = 10; // 2^n, n >= 5 
    unsigned int seed = 0;
    long long size = length * length;
    int threads_per_block = std::min(1024LL, size);
    int blocks = size/threads_per_block;
    long long warm_up_steps = 1024;
    long long n_itter = warm_up_steps/size;

    // --------------------------------------------------------------------------------------------
    curandState_t* states; // used to store random state for each core
    cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
    init_rand<<<blocks, threads_per_block>>>(seed, states); // initialize for all states
    
    //std::cout << "init_rand completed" << std::endl;
    
    // allocate memory in device
    float* d_spins; // device copy of a spin system (pp_spins[i])
    int* d_length; // device copy of length
    long long* d_n_itter; // device copy of n_itter
    cudaMalloc((void**)&d_spins, size * sizeof(float));
    cudaMalloc((void**)&d_length, sizeof(int));
    cudaMalloc((void**)&d_n_itter, sizeof(long long));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_itter, &n_itter, sizeof(long long), cudaMemcpyHostToDevice);

    float* p_spins = new float[size];
    cudaDeviceSynchronize();

    
    for (long long i = 0; i < n_sample; ++i) {
       // initialize spins
        initialize <<<blocks, threads_per_block>>> (d_spins, states);

        // warm up 
        warm_up <<<blocks, threads_per_block>>> (d_spins, d_length, d_n_itter);
        
        // copy memory from device to host
        cudaMemcpy(p_spins, d_spins, size * sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << i << std::endl;
        cudaDeviceSynchronize();

    }
    
    print_spins (p_spins, length, std::min(10, length), std::min(15,length));

    // -----------------------------------------------------------------
    cudaFree(&d_length); cudaFree(d_spins);
    cudaFree(states); 
    delete[] p_spins;
    return 0;
}
