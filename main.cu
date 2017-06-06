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
void warm_up (float* spins, int* p_length, long long* p_n_itter, curandState_t* states) {
    long long n_itter = *p_n_itter;
    int length = *p_length;
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int dim1, dim2;
    float upper_spin, lower_spin, left_spin, right_spin;
    float current_energy, changed_energy;
    float new_spin, current_spin;
    for (long long i = 0; i < n_itter; ++i){
        dim1 = 0;
        dim2 = 0;

        // get current position spin
        current_spin = spins[dim1 * length + dim2];

        // get upper spin
        if (dim1 == 0) {
            upper_spin = spins[(length-1) * length + dim2];
        } else {
            upper_spin = spins[(dim1-1) * length + dim2];
        }

        // get lower spin
        if (dim1 == length-1) {
            lower_spin = spins[0 * length + dim2];
        } else {
            lower_spin = spins[(dim1+1) * length + dim2];
        }

        // get left spin
        if (dim2 == 0) {
            left_spin = spins[dim1 * length + (length-1)];
        } else {
            left_spin = spins[dim1 * length + (dim2-1)];
        }

        // get right spin
        if (dim2 == length-1) {
            right_spin = spins[dim1 * length + 0];
        } else {
            right_spin = spins[dim1 * length + (dim2+1)];
        }

        // get energy
        current_energy = -(std::cos(current_spin - upper_spin) + std::cos(current_spin - lower_spin)
                           + std::cos(current_spin - left_spin) + std::cos(current_spin - right_spin));

        // new spin direction
        new_spin = curand_uniform(&states[index]) * 2*M_PI;

        spins[0] = current_spin;
        spins[1] = upper_spin;
        spins[2] = lower_spin;
        spins[3] = left_spin;
        spins[4] = right_spin;
        spins[5] = std::cos(current_spin - upper_spin);
        spins[6] = std::cos(current_spin - lower_spin);
        spins[7] = std::cos(current_spin - left_spin);
        spins[8] = std::cos(current_spin - right_spin);
        spins[9] = current_energy;
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
    long long n_sample = 1;
    int length = 10; // 2^n, n >= 5 
    unsigned int seed = 0;
    long long size = length * length;
    int threads_per_block = std::min(1024LL, size);
    int blocks = size/threads_per_block;
    long long warm_up_steps = 1024;
    //long long n_itter = warm_up_steps/size;
    long long n_itter = 1;
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
        //warm_up <<<blocks, threads_per_block>>> (d_spins, d_length, d_n_itter, states);
        warm_up <<<1,1>>> (d_spins, d_length, d_n_itter, states);

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
