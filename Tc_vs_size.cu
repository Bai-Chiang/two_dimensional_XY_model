//--- file: Tc_vs_size.cu ---



#include <cstdio>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <string>
#include "src/xy_model.h"
#include "src/cuda_reduction.h"


int main() {
    // some values can be adjusted
    float T_min = 0.0, T_max = 5.0; // temperature range: (T_min, T_max]  suppose boltzmann constant k = 1
    int n_T = 200; // number of temperature interpolate pints
    long long n_sample_1024= 2500LL; // number of sample for length = 1024, if length is small needs more warm up steps

    // ============================================================================================

    long long n_samples[] = {n_sample_1024*512, n_sample_1024*256, n_sample_1024*64, n_sample_1024*16, n_sample_1024*4, n_sample_1024};

    // some variables used in later calculation
    int n_length = 6;
    int lengths[] = {32, 64, 128, 256, 512, 1024}; // all valus should be 2^n, 6 <= n <= 10    length of 2D-spins, the 2D-spins lattice will be length * length
    long long warm_up_steps; // length * length * 2^n    warm up step is proportional to total number of spins
    int length;
    long long size;
    unsigned int seed;
    int threads_per_block;
    int blocks;
    long long n_itter;
    long long n_sample = n_samples[0];

    float T;
    double* E_sample = new double[n_sample*n_T];
    double* specific_heat_per_spin_sample = new double[n_sample*n_T];

    // device variables
    curandState_t* states; // used to store random state for each core
    float* d_spins; // a 2D-spin recording each spin's direction theta
    int* d_length; // length
    long long* d_n_itter; // n_itter
    float* d_T; // T
    float* d_E; // energy for each spin: E_i = -cos(theta_i - theta_j) for adjacent j
    // used to sotre output array
    float* d_oE;
    // when reducing 2D-spin need these variable
    long long reduced_length;
    float E[1024];
    long long index;
    long long index_next, index_prev;

    // variables used in analysis results
    double* specific_heat_per_spin_vs_T = new double[n_length * n_T]; // C_v = ( <E^2> - <E>^2 )/T^2
    double* E_mean = new double[n_T];
    double* E_per_spin_vs_T = new double[n_T];
    double f_sample;
    double* Tc_vs_size = new double[n_length];
    double Tc;
    double Cv_max;


    for (int i_length = 0; i_length < n_length; ++i_length) {
        length = lengths[i_length];
        warm_up_steps = (long long)(length * length) * 8;
        n_sample = n_samples[i_length];
        f_sample = double(n_sample);

        // host
        size = length * length; // the total size of 2D-spin lattice is length * length
        seed = time(NULL); // seed of random numbers
        threads_per_block = std::min(1024, length);
        blocks = size/threads_per_block;
        n_itter = warm_up_steps/size; // sice total number of threads is size (threads_per_block * blocks), to obtain warm_up_steps, need to itterate warm_up_steps/size times

        // device
        cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
        init_rand<<<blocks, threads_per_block>>>(seed, states); // initialize for all states

        cudaMalloc((void**)&d_spins, size * sizeof(float));
        cudaMalloc((void**)&d_length, sizeof(int));
        cudaMalloc((void**)&d_n_itter, sizeof(long long));
        cudaMalloc((void**)&d_T, sizeof(float));
        cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_itter, &n_itter, sizeof(long long), cudaMemcpyHostToDevice);


        // ============================================================================================
        // calculate and restore the result for each sample at each T
        for (long long i_sample = 0; i_sample < n_sample; ++i_sample) {

            std::cout << "length: " << lengths[i_length] << "    i sample: " << i_sample << std::endl;

            // initialize spins
            initialize <<<blocks, threads_per_block>>> (d_spins, states);

            // To get to stable state, first warm up needs more steps
            for (long long i = 0; i < n_itter * 5 ; ++i) {
                // only needs half of length, so at this time # of threads in a block is half of previous
                warm_up_type_1<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states);
                warm_up_type_2<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states);
            }

            for (int i_T = n_T-1; i_T >= 0; --i_T ) { // temperature goes down

                T = T_min + float(i_T+1) * (T_max - T_min)/float(n_T);
                cudaMemcpy(d_T, &T, sizeof(float), cudaMemcpyHostToDevice);

                // warm up
                for (long long i = 0; i < n_itter; ++i) {
                    // only needs half of length, so at this time # of threads in a block is half of previous
                    warm_up_type_1<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states);
                    warm_up_type_2<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states);
                }

                // get energy
                cudaMalloc((void**)&d_E, size*sizeof(float));
                get_energy<<<blocks, threads_per_block>>> (d_E, d_spins, d_length);

                // reduce E
                reduced_length = size;
                while (reduced_length > 1024LL) {
                    reduced_length /= 1024LL;
                    cudaMalloc((void**)&d_oE, reduced_length * sizeof(float));
                    cuda_reduce <<< reduced_length, 512, 512*sizeof(float) >>> (d_E, d_oE);
                    cudaFree(d_E);
                    cudaMalloc((void**)&d_E, reduced_length * sizeof(float));
                    cudaMemcpy(d_E, d_oE, reduced_length * sizeof(float), cudaMemcpyDeviceToDevice);
                    cudaFree(d_oE);
                }
                cudaMemcpy(E, d_E, reduced_length * sizeof(float), cudaMemcpyDeviceToHost);
                cudaFree(d_E);

                index = i_T * n_sample + i_sample;
                E_sample[index] = 0.0;
                for (int i = 0; i < reduced_length; ++i) {
                    /* ----------------------------------------------------
                     * there is a novel bug
                     * for {64, 128, 256, 512, 1024}, the last length 1024
                     * E[1023] = nan
                     * but if i tried {32, 32, 32, 32, 1024}
                     * this bug will not appear
                     * so I add this criteria : !isnan(E[i])
                     * need to be solve later
                     */
                    if (!isnan(E[i])){
                        E_sample[index] += E[i];
                    }
                }
            }
        }

        cudaFree(d_spins);
        cudaFree(d_length);
        cudaFree(d_n_itter);
        cudaFree(d_T);
        cudaFree(states);


        // ============================================================================================
        // analysis result

        // get specific_heat_per_spin_sample
        for (int i_sample = 0; i_sample < n_sample; ++i_sample) {
           for (int i_T = 0; i_T < n_T; ++i_T) {
               index = i_T * n_sample + i_sample;
               if (i_T == 0) {
                   index_next = (i_T + 1) * n_sample + i_sample;
                   specific_heat_per_spin_sample[index] = (E_sample[index_next] -  E_sample[index])/( double(T_max - T_min)/double(n_T) );
               } else if (i_T == n_T-1) {
                   index_prev = (i_T - 1) * n_sample + i_sample;
                   specific_heat_per_spin_sample[index] = (E_sample[index] -  E_sample[index_prev])/( double(T_max - T_min)/double(n_T) );
               } else {
                   index_next = (i_T + 1) * n_sample + i_sample;
                   index_prev = (i_T - 1) * n_sample + i_sample;
                   specific_heat_per_spin_sample[index] = (E_sample[index_next] -  E_sample[index_prev])/( double(T_max - T_min)*2.0/double(n_T) );
               }
           }
        }

        for (int i_T = 0; i_T < n_T; ++i_T) {
            E_mean[i_T] = 0.0;
            specific_heat_per_spin_vs_T[i_length * n_T + i_T] = 0.0;
            for (long long i_sample = 0; i_sample < n_sample; ++i_sample) {
                index = i_T * n_sample + i_sample;
                specific_heat_per_spin_vs_T[i_length * n_T + i_T] += specific_heat_per_spin_sample[index]/f_sample;
            }
            specific_heat_per_spin_vs_T[i_length * n_T + i_T] = specific_heat_per_spin_vs_T[i_length * n_T + i_T]/double(size);
        }


        // find Tc. Tc is when Cv get maximum value
        Tc = 0; Cv_max = 0;
        for (int i_T = 0; i_T < n_T; ++i_T){
            index = i_length * n_T + i_T;
            if (specific_heat_per_spin_vs_T[index] > Cv_max) {
                Cv_max = specific_heat_per_spin_vs_T[index];
                Tc = double(T_min) + double( i_T + 1) * (T_max - T_min)/double(n_T);
            }
        }
        Tc_vs_size[i_length] = Tc;
    }

    // --------------------------------------------------------------------------------------------
    // write file

    std::string str_sample = std::to_string(n_sample);
    std::string str_n_warm = std::to_string(n_itter);
    std::string file_name = std::string("result/Tc_vs_size/") + str_sample + std::string(".data");

    FILE* pfile;
    pfile = fopen(file_name.c_str(), "w");
    if (pfile != NULL) {
        fprintf(pfile, "%lld\n", n_itter);
        for(int i = 0; i < n_length; ++i) {
            fprintf(pfile, "%d,", lengths[i]);
        }
        fprintf(pfile, "\n" );
        for(int i = 0; i < n_length;++i) {
            fprintf(pfile, "%lld,", n_samples[i]);
        }
        fprintf(pfile, "\n" );
        for(int i = 0; i < n_length; ++i) {
            fprintf(pfile, "%.10f,", Tc_vs_size[i]);
        }
        fprintf(pfile, "\n" );
        for(int i = 0; i < n_T; ++i) {
            T = T_min + float(i+1) * (T_max - T_min)/float(n_T);
            fprintf(pfile, "%.10f,", T);
        }
        fprintf(pfile, "\n");
        for (int i_length = 0; i_length < n_length; ++i_length){
            for (int i_T = 0; i_T < n_T; ++i_T) {
                fprintf(pfile, "%.10f,", specific_heat_per_spin_vs_T[i_length * n_T + i_T]);
            }
            if (i_length != n_length -1){
                fprintf(pfile, "\n");
            }
        }

    } else {
        printf("ERROR, unable to open file 'result/spins.csv' !");
    }
    fclose(pfile);

    // -----------------------------------------------------------------
    delete[] specific_heat_per_spin_vs_T;
    delete[] E_mean;
    delete[] E_per_spin_vs_T;
    delete[] Tc_vs_size;

    return 0;
}
