// --- file: magnetisation_susceptibility_specific_heat_vs_temperature.cu ---




#include <cstdio>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <string>
#include "src/xy_model.h"


int main() {

    // some values can be adjusted
    int length = 128; // 2^n, n >= 5    length of 2D-spins, the 2D-spins lattice will be length * length
    unsigned int seed = 0; // seed of random numbers    set seed to a fix number, so that each time you run will get same result
    long long warm_up_steps = length * length * 32; // length * length * 2^n    warm up step is proportional to total number of spins
    float T_min = 0.001, T_max = 2.0; // temperature range, suppose boltzmann constant k = 1
    int n_T = 1000; // number of temperature interpolate pints, include T_min and T_max
    long long n_sample = 1; // plot one figure of 2D-spins, hence n_sample = 1



    // ============================================================================================
    long long size = length * length; // the total size of 2D-spin lattice is length * length
    int threads_per_block = std::min(1024LL, size);
    int blocks = size/threads_per_block;
    long long n_itter = warm_up_steps/size; // sice total number of threads is size (threads_per_block * blocks), to obtain warm_up_steps, need to itterate warm_up_steps/size times
    float T;
    float* p_spins = new float[size];
    float* Mx_vs_T, My_vs_T; // Mx = sum(Sx)/n_sample    My = sum(Sy)/n_sample
    float* susceptibility_vs_T; // chi = ( <|M|^2> - <|M|>^2 )/T
    float* specific_heat_vs_T; // C_v = ( <E^2> - <E>^2 )/T^2
    
    // --------------------------------------------------------------------------------------------
    curandState_t* states; // used to store random state for each core
    cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
    init_rand<<<blocks, threads_per_block>>>(seed, states); // initialize for all states


    // allocate memory in device
    float* d_spins; // a 2D-spin recording each spin's direction theta
    int* d_length; // length
    long long* d_n_itter; // n_itter
    float* d_T; // T
    float* d_Sx, d_Sy; // x or y component of spin
    float* d_Mx_n_sample, d_My_n_sample; // used to record result of each sample
    float* d_M_n_sample_squared, d_M_n_sample; // |M|^2 and |M|
    float* d_E_n_sample_squared, d_E_n_sample; // E^2 and E

    cudaMalloc((void**)&d_spins, size * sizeof(float));
    cudaMalloc((void**)&d_length, sizeof(int));
    cudaMalloc((void**)&d_n_itter, sizeof(long long));
    cudaMalloc((void**)&d_T, sizeof(float));
    cudaMalloc((void**)&d_Sx, size*sizeof(float));
    cudaMalloc((void**)&d_Sy, size*sizeof(float));
    cudaMalloc((void**)&d_Mx_n_sample, n_sample*sizeof(float));
    cudaMalloc((void**)&d_My_n_sample, n_sample*sizeof(float));
    //cudaMalloc((void**)&d_Mx_vs_T, n_T*sizeof(float));
    //cudaMalloc((void**)&d_My_vs_T, n_T*sizeof(float));
    cudaMalloc((void**)&d_M_n_sample_squared, n_sample*sizeof(float));
    cudaMalloc((void**)&d_M_n_sample, n_sample*sizeof(float));
    //cudaMalloc((void**)&d_susceptibility_vs_T, n_T*sizeof(float));
    cudaMalloc((void**)&d_E_n_sample_squared, n_sample*sizeof(float));
    cudaMalloc((void**)&d_E_n_sample, n_sample*sizeof(float));
    //cudaMalloc((void**)&d_specific_heat_vs_T, n_T*sizeof(float));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_itter, &n_itter, sizeof(long long), cudaMemcpyHostToDevice);
    
    
    for (long long i_sample = 0; i_sample < n_sample; ++i_sample) {

        // initialize spins
        initialize <<<blocks, threads_per_block>>> (d_spins, states);

        for (int i_T = n_T; i_T >= 0; --i_T ) { // temperature goes down
            
            T = T_min + i_T * (T_max - T_min)/float(n_T);
            cudaMemcpy(d_T, &T, sizeof(float), cudaMemcpyHostToDevice);

            // warm up
            warm_up <<<blocks, threads_per_block>>> (d_spins, d_T, d_length, d_n_itter, states);

        }
    }

    // copy memory from device to host
    cudaMemcpy(p_spins, d_spins, size * sizeof(float), cudaMemcpyDeviceToHost);

    // --------------------------------------------------------------------------------------------
    // write file
/*
    std::string str_L = std::to_string(length);
    std::string str_T = std::to_string(T);
    std::string str_n_warm = std::to_string(warm_up_steps);
    std::string file_name = std::string("result/vortex_configuration/") + str_L + std::string("_") + str_T + "_" + str_n_warm + std::string(".data");
    
    FILE* pfile;
    pfile = fopen(file_name.c_str(), "w");
    if (pfile != NULL) {
        fprintf(pfile, "%d\n%f\n%lld\n", length, T, warm_up_steps);
        for (int i = 0; i < length; ++i) {
            for (int j = 0; j < length; ++j) {
                fprintf(pfile, "%f,", p_spins[i*length + j]);
            }
            fprintf(pfile, "\n");
        }
    } else {
        printf("ERROR, unable to open file 'result/spins.csv' !");
    }
    fclose(pfile);
*/
    // -----------------------------------------------------------------
    cudaFree(&d_length); cudaFree(d_spins);
    cudaFree(states);
    delete[] p_spins;

    return 0;
}

