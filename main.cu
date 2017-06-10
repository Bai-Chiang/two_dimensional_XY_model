//
// Created by Baiqiang Qiang on 30/05/2017.
//

#include <cstdio>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>

#include "src/xy_model.h"


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
    //clock_t begin = std::clock();

    long long n_sample = std::pow(2,0);
    int length = 128; // 2^n, n >= 5
    unsigned int seed = 0;
    long long size = length * length;
    int threads_per_block = std::min(1024LL, size);
    int blocks = size/threads_per_block;
    long long warm_up_steps = size * 32LL;
    long long n_itter = warm_up_steps/size;
    //long long n_itter = 1;
    float T = 0.5;
    // --------------------------------------------------------------------------------------------
    curandState_t* states; // used to store random state for each core
    cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
    init_rand<<<blocks, threads_per_block>>>(seed, states); // initialize for all states

    //std::cout << "init_rand completed" << std::endl;

    // allocate memory in device
    float* d_spins; // device copy of a spin system (pp_spins[i])
    int* d_length; // device copy of length
    long long* d_n_itter; // device copy of n_itter
    float* d_T; // device copy of T
    cudaMalloc((void**)&d_spins, size * sizeof(float));
    cudaMalloc((void**)&d_length, sizeof(int));
    cudaMalloc((void**)&d_n_itter, sizeof(long long));
    cudaMalloc((void**)&d_T, sizeof(float));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_itter, &n_itter, sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, &T, sizeof(float), cudaMemcpyHostToDevice);

    float* p_spins = new float[size];

    for (long long i = 0; i < n_sample; ++i) {

        // initialize spins
        initialize <<<blocks, threads_per_block>>> (d_spins, states);

        //cudaMemcpy(p_spins, d_spins, size * sizeof(float), cudaMemcpyDeviceToHost);
        //print_spins (p_spins, length, 30, 15);

        // warm up
        warm_up <<<blocks, threads_per_block>>> (d_spins, d_T, d_length, d_n_itter, states);
        //warm_up <<<1,1>>> (d_spins, d_T, d_length, d_n_itter, states);

    }

    // copy memory from device to host
    cudaMemcpy(p_spins, d_spins, size * sizeof(float), cudaMemcpyDeviceToHost);
    print_spins (p_spins, length, std::min(30, length), std::min(15, length));

    // write file
    FILE* pfile;
    pfile = fopen("result/quiver_plot_T_0.5.csv", "w");
    if (pfile != NULL) {
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

    // -----------------------------------------------------------------
    cudaFree(&d_length); cudaFree(d_spins);
    cudaFree(states);
    delete[] p_spins;

    return 0;
}
