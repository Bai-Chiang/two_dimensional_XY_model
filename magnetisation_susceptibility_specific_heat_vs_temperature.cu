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
#include "src/cuda_reduction.h"

void print (float* arr, int length) {
    for (long long i = 0; i < length; ++i) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}
void print (double* arr, int length) {
    for (long long i = 0; i < length; ++i) {
        printf("%lf ", arr[i]);
    }
    printf("\n");
}


void print_arr (float* arr, int dim1, int dim2) {
    for (int i = 0; i < dim1; ++i){
        for (int j = 0; j < dim2; ++j) {
            printf("%.3f ", arr[i * dim2 + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {

    // some values can be adjusted
    int length = 512; // 2^n, n >= 5    length of 2D-spins, the 2D-spins lattice will be length * length
    long long warm_up_steps = (long long)(length * length) * 8; // length * length * 2^n    warm up step is proportional to total number of spins
    float T_min = 0.0, T_max = 5.0; // temperature range: (T_min, T_max]  suppose boltzmann constant k = 1
    int n_T = 200; // number of temperature interpolate pints, T_min < T <= T_max. If n_T = 100, T_min = 0.0, T_max = 2.0, then T = 0.02, 0.04, ..., 1.98, 2.00
    long long n_sample = 100000LL;


    // ============================================================================================
    // some variables used in later calculation

    // ------------------------------------------
    // host part
    unsigned int seed = time(NULL); // seed of random numbers
    long long size = length * length; // the total size of 2D-spin lattice is length * length
    int threads_per_block = std::min(1024, length);
    int blocks = size/threads_per_block;
    long long n_itter = warm_up_steps/size; // sice total number of threads is size (threads_per_block * blocks), to obtain warm_up_steps, need to itterate warm_up_steps/size times
    float T;
    //float* p_spins = new float[size];
    double* Mx_sample = new double[n_sample*n_T]; //used to record result of each sample at each T
    double* My_sample = new double[n_sample*n_T];
    double* M_squared_sample = new double[n_sample*n_T];
    double* E_sample = new double[n_sample*n_T];
    double* E_squared_sample = new double[n_sample*n_T];
    double* specific_heat_per_spin_sample = new double[n_sample*n_T];

    // when reducing 2D-spin need these variable
    long long reduced_length;
    float Sx[1024];
    float Sy[1024];
    float E[1024];
    long long index;
    long long index_next, index_prev;
//float* debug_spin = new float[size];
//float* debug_Sx = new float[size];

    // ------------------------------------------
    // device part
    curandState_t* states; // used to store random state for each core
    cudaMalloc((void**) &states, size * sizeof(curandState_t)); // allocate memory in device
    init_rand<<<blocks, threads_per_block>>>(seed, states); // initialize for all states


    // allocate memory in device
    float* d_spins; // a 2D-spin recording each spin's direction theta
    int* d_length; // length
    long long* d_n_itter; // n_itter
    float* d_T; // T
    float* d_Sx; // x component of spin
    float* d_Sy; // y component of spin
    float* d_E; // energy for each spin: E_i = -cos(theta_i - theta_j) for adjacent j
    //float* d_Mx_n_sample, d_My_n_sample; // used to record result of each sample
    //float* d_M_n_sample_squared, d_M_n_sample; // |M|^2 and |M|
    //float* d_E_n_sample_squared, d_E_n_sample; // E^2 and E
    // used to sotre output array
    float* d_oSx;
    float* d_oSy;
    float* d_oE;

    cudaMalloc((void**)&d_spins, size * sizeof(float));
    cudaMalloc((void**)&d_length, sizeof(int));
    cudaMalloc((void**)&d_n_itter, sizeof(long long));
    cudaMalloc((void**)&d_T, sizeof(float));
    cudaMemcpy(d_length, &length, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n_itter, &n_itter, sizeof(long long), cudaMemcpyHostToDevice);


    // ============================================================================================
    // calculate and restore the result for each sample at each T
    for (long long i_sample = 0; i_sample < n_sample; ++i_sample) {

        std::cout << "i sample: " << i_sample << std::endl;

        // initialize spins
        initialize <<<blocks, threads_per_block>>> (d_spins, states);

        // To get to stable state, first warm up needs more steps
        for (long long i = 0; i < n_itter*5; ++i) {
            // only needs half of length, so at this time # of threads in a block is half of previous
            warm_up_type_1<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states); 
            warm_up_type_2<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states);
        }
        
        for (int i_T = n_T-1; i_T >= 0; --i_T ) { // temperature goes down

            T = T_min + float(i_T+1) * (T_max - T_min)/float(n_T);
            cudaMemcpy(d_T, &T, sizeof(float), cudaMemcpyHostToDevice);

            //std::cout << "i sample: " << i_sample << "    T: " << T << std::endl;

            // warm up
            //warm_up <<<blocks, threads_per_block>>> (d_spins, d_T, d_length, d_n_itter, states);
            
            for (long long i = 0; i < n_itter; ++i) {
                // only needs half of length, so at this time # of threads in a block is half of previous
                warm_up_type_1<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states); 
                warm_up_type_2<<<blocks, threads_per_block/2>>> (d_spins, d_T, d_length, states);
            }
            

            // get spin
            cudaMalloc((void**)&d_Sx, size*sizeof(float));
            cudaMalloc((void**)&d_Sy, size*sizeof(float));
            get_spin <<<blocks, threads_per_block>>> (d_Sx, d_Sy, d_spins);

            // get energy
            cudaMalloc((void**)&d_E, size*sizeof(float));
            get_energy<<<blocks, threads_per_block>>> (d_E, d_spins, d_length);

            // reduce Sx, Sy, E
            reduced_length = size;
            while (reduced_length > 1024LL) {
                reduced_length /= 1024LL;
                cudaMalloc((void**)&d_oSx, reduced_length * sizeof(float));
                cudaMalloc((void**)&d_oSy, reduced_length * sizeof(float));
                cudaMalloc((void**)&d_oE, reduced_length * sizeof(float));
                cuda_reduce <<< reduced_length, 512, 512*sizeof(float) >>> (d_Sx, d_oSx);
                cuda_reduce <<< reduced_length, 512, 512*sizeof(float) >>> (d_Sy, d_oSy);
                cuda_reduce <<< reduced_length, 512, 512*sizeof(float) >>> (d_E, d_oE);
                cudaFree(d_Sx);
                cudaFree(d_Sy);
                cudaFree(d_E);
                cudaMalloc((void**)&d_Sx, reduced_length * sizeof(float));
                cudaMalloc((void**)&d_Sy, reduced_length * sizeof(float));
                cudaMalloc((void**)&d_E, reduced_length * sizeof(float));
                cudaMemcpy(d_Sx, d_oSx, reduced_length * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_Sy, d_oSy, reduced_length * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_E, d_oE, reduced_length * sizeof(float), cudaMemcpyDeviceToDevice);
                cudaFree(d_oSx);
                cudaFree(d_oSy);
                cudaFree(d_oE);
            }
            cudaMemcpy(Sx, d_Sx, reduced_length * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(Sy, d_Sy, reduced_length * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(E, d_E, reduced_length * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_Sx);
            cudaFree(d_Sy);
            cudaFree(d_E);

            index = i_T * n_sample + i_sample;
            Mx_sample[index] = 0.0;
            My_sample[index] = 0.0;
            E_sample[index] = 0.0;
            for (int i = 0; i < reduced_length; ++i) {
                Mx_sample[index] += Sx[i]/double(size);
                My_sample[index] += Sy[i]/double(size);
                E_sample[index] += E[i];
                //M_squared_sample[index] += Mx_sample[index] * Mx_sample[index] + My_sample[index] * My_sample[index];
                //E_squared_sample[index] += E[i] * E[i];
            }
        }
    }

    cudaFree(d_spins);
    cudaFree(d_length);
    cudaFree(d_n_itter);
    cudaFree(d_T);
    //cudaFree(d_Sx);
    //cudaFree(d_Sy);
    //cudaFree(d_E);


    // ============================================================================================
    // analysis result

    double* Mx_vs_T = new double[n_T]; double* My_vs_T = new double[n_T]; // Mx = sum(Sx)/n_sample    My = sum(Sy)/n_sample
    double* susceptibility_vs_T = new double[n_T]; // chi = ( <|M|^2> - <|M|>^2 )/T
    double* specific_heat_per_spin_vs_T = new double[n_T]; // C_v = ( <E^2> - <E>^2 )/T^2
    double* E_per_spin_vs_T = new double[n_T];

    double* M_mean_square = new double[n_T];
    double* M_mean = new double[n_T];
    double* E_mean = new double[n_T];
    //double* E_mean_square = new double[n_T];

    double f_sample = double(n_sample);
    double T_double;

    // get specific_heat_per_spin_sample
    for (int i_sample = 0; i_sample < n_sample; ++i_sample) {
       for (int i_T = 0; i_T < n_T; ++i_T) {
           index = i_T * n_sample + i_sample;
           //std::cout << "index: " << index << std::endl;
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
        M_mean[i_T] = 0.0;
        M_mean_square[i_T] = 0.0;
        E_mean[i_T] = 0.0;
        //E_mean_square[i_T] = 0.0;
        Mx_vs_T[i_T] = 0.0;
        My_vs_T[i_T] = 0.0;
        susceptibility_vs_T[i_T] = 0.0;
        specific_heat_per_spin_vs_T[i_T] = 0.0;
        E_per_spin_vs_T[i_T] = 0.0;
        for (long long i_sample = 0; i_sample < n_sample; ++i_sample) {
            index = i_T * n_sample + i_sample;
            M_mean[i_T] += std::sqrt(Mx_sample[index]*Mx_sample[index] + My_sample[index]*My_sample[index]) / f_sample;
            M_mean_square[i_T] += (Mx_sample[index] * Mx_sample[index] + My_sample[index] * My_sample[index])/f_sample;
            E_mean[i_T] += E_sample[index]/f_sample;
            specific_heat_per_spin_vs_T[i_T] += specific_heat_per_spin_sample[index]/f_sample;
            //E_mean_square[i_T] += E_sample[index] * E_sample[index]/f_sample;
            Mx_vs_T[i_T] += Mx_sample[index]/f_sample;
            My_vs_T[i_T] += My_sample[index]/f_sample;
        }
        T_double = double( T_min + double(i_T+1) * (T_max - T_min)/double(n_T) );
        susceptibility_vs_T[i_T] = (M_mean_square[i_T] - M_mean[i_T]*M_mean[i_T])/T_double;
        E_per_spin_vs_T[i_T] = E_mean[i_T]/double(size);
        specific_heat_per_spin_vs_T[i_T] = specific_heat_per_spin_vs_T[i_T]/double(size);
        //print(specific_heat_per_spin_vs_T, n_T);
    }
    
    /*
    std::cout<< "Mx:\n";
    print(Mx_vs_T, n_T);
    std::cout << std::endl;
    std::cout << "My:\n";
    print(My_vs_T, n_T);
    std::cout<< std::endl;
    std::cout << "E:\n";
    print(E_mean, n_T);
    std::cout<< std::endl;
    std::cout << "Chi:\n";
    print(susceptibility_vs_T,n_T);
    std::cout << std::endl;
    */
    //std::cout << "Cv:\n";
    //print(specific_heat_per_spin_vs_T, n_T);
    //print(specific_heat_per_spin_sample, n_T);



    // --------------------------------------------------------------------------------------------
    // write file

    std::string str_L = std::to_string(length);
    std::string str_sample = std::to_string(n_sample);
    std::string str_n_warm = std::to_string(n_itter);
    std::string file_name = std::string("result/magnetisation_susceptibility_specific_heat_vs_temperature/") + str_L + std::string("_") + str_sample + "_" + str_n_warm + std::string(".data");

    FILE* pfile;
    pfile = fopen(file_name.c_str(), "w");
    if (pfile != NULL) {
        fprintf(pfile, "%d\n%lld\n%lld\n", length, n_sample, n_itter);
        for(int i = 0; i < n_T; ++i) {
            T = T_min + float(i+1) * (T_max - T_min)/float(n_T);
            fprintf(pfile, "%.10f,", T);
        }
        fprintf(pfile, "\n");
        for (int i = 0; i < n_T; ++i) {
            fprintf(pfile, "%.10f,", Mx_vs_T[i]);
        }
        fprintf(pfile, "\n");
        for (int i = 0; i < n_T; ++i) {
            fprintf(pfile, "%.10f,", My_vs_T[i]);
        }
        fprintf(pfile, "\n");
        for (int i = 0; i < n_T; ++i) {
            fprintf(pfile, "%.10f,", M_mean[i]);
        }
        fprintf(pfile, "\n");
        for (int i = 0; i < n_T; ++i) {
            fprintf(pfile, "%.10f,", susceptibility_vs_T[i]);
        }
        fprintf(pfile, "\n");
        for (int i = 0; i < n_T; ++i) {
            fprintf(pfile, "%.10f,", E_per_spin_vs_T[i]);
        }
        fprintf(pfile, "\n");
        for (int i = 0; i < n_T; ++i) {
            fprintf(pfile, "%.10f,", specific_heat_per_spin_vs_T[i]);
        }

    } else {
        printf("ERROR, unable to open file 'result/spins.csv' !");
    }
    fclose(pfile);

    // -----------------------------------------------------------------
    //cudaFree(&d_length); cudaFree(d_spins);
    //cudaFree(states);
    //delete[] p_spins;

    return 0;
}
