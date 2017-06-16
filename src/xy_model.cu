#include <cmath>
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>
#include <cstdio>

#include "xy_model.h"


/* ------------------------------------------------------------------------------------------------
 * initize seed and sequence for all device cores
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
__global__ void initialize (float* spins, curandState_t* states) {
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    spins[index] = curand_uniform(&states[index]) * 2*M_PI;
}




/* ------------------------------------------------------------------------------------------------
 * warm up the system
 * n_itter is the itteration times for each thread
 * the total warm_up_step for the whole system is n_itter * n_block * n_threads_per_block
 */
__global__ void warm_up (float* spins, float* p_T, int* p_length, long long* p_n_itter, curandState_t* states) {
    long long n_itter = *p_n_itter;
    int length = *p_length;
    float T = *p_T;
    int index = blockIdx.x *blockDim.x + threadIdx.x;
    int dim1, dim2;
    float upper_spin, lower_spin, left_spin, right_spin;
    float current_energy, changed_energy, delta_energy;
    float new_spin, current_spin;
    for (long long i = 0; i < n_itter; ++i){
        dim1 = curand(&states[index])%length;
        dim2 = curand(&states[index])%length;

        // get current position spin
        current_spin = spins[dim1 * length + dim2];

        // get upper spin
        if (dim1 != 0) {
            upper_spin = spins[(dim1-1) * length + dim2];
        } else {
            upper_spin = spins[(length-1) * length + dim2];
        }

        // get lower spin
        if (dim1 != length-1) {
            lower_spin = spins[(dim1+1) * length + dim2];
        } else {
            lower_spin = spins[0 * length + dim2];
        }

        // get left spin
        if (dim2 != 0) {
            left_spin = spins[dim1 * length + (dim2-1)];
        } else {
            left_spin = spins[dim1 * length + (length-1)];
        }

        // get right spin
        if (dim2 != length-1) {
            right_spin = spins[dim1 * length + (dim2+1)];
        } else {
            right_spin = spins[dim1 * length + 0];
        }

        // get energy
        current_energy = -(std::cos(current_spin - upper_spin) + std::cos(current_spin - lower_spin)
                           + std::cos(current_spin - left_spin) + std::cos(current_spin - right_spin));

        // new spin direction
        new_spin = curand_uniform(&states[index]) * 2*M_PI;

        // changed energy
        changed_energy = -(std::cos(new_spin - upper_spin) + std::cos(new_spin - lower_spin)
                           + std::cos(new_spin - left_spin) + std::cos(new_spin - right_spin));

        // decide whether change spin
        delta_energy = changed_energy - current_energy;
        if ( curand_uniform(&states[index]) < std::exp(-delta_energy/T) ) {
            spins[dim1 * length + dim2] = new_spin;
        }

    }
}



/* ------------------------------------------------------------------------------------------------
 * get x and y component of spins
 */
__global__ void get_spin(float* Sx, float* Sy, float* theta) {
    long long global_id = blockIdx.x *blockDim.x + threadIdx.x;
    Sx[global_id] = std::cos(theta[global_id]);
    Sy[global_id] = std::sin(theta[global_id]);
}


/* ------------------------------------------------------------------------------------------------
 * get system energy for each spin
 */
__global__ void get_energy (float* energy, float* spins, int* p_length) {
    int length = *p_length;
    long long global_id = blockIdx.x * blockDim.x + threadIdx.x;
    float current_spin, upper_spin, lower_spin, left_spin, right_spin;

    // get current position spin
    current_spin = spins[blockIdx.x * length + threadIdx.x];

    // get upper spin
    if (blockIdx.x == 0) {
        upper_spin = spins[(length-1) * length + threadIdx.x];
    } else {
        upper_spin = spins[(blockIdx.x-1) * length + threadIdx.x];
    }

    // get lower spin
    if (blockIdx.x == length) {
        lower_spin = spins[0 * length + threadIdx.x];
    } else {
        lower_spin = spins[(blockIdx.x+1) * length + threadIdx.x];
    }

    // get left spin
    if (threadIdx.x != 0) {
        left_spin = spins[blockIdx.x * length + (threadIdx.x-1)];
    } else {
        left_spin = spins[blockIdx.x * length + (length-1)];
    }

    // get right spin
    if (threadIdx.x != length-1) {
        right_spin = spins[blockIdx.x * length + (threadIdx.x+1)];
    } else {
        right_spin = spins[blockIdx.x * length + 0];
    }

    // get energy
    energy[global_id] = -(std::cos(current_spin - upper_spin) + std::cos(current_spin - lower_spin)
                          + std::cos(current_spin - left_spin) + std::cos(current_spin - right_spin));
}

/* ------------------------------------------------------------------------------------------------
 * warm up the system
 * warm up type 1 will warm up these kind of points. 1 means warm up, 0 means not warm up.
 * each 1 will be warm up one time
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   . . . . . . . . .
 *   . . . . . . . .  .
 *   . . . . . . . .    .
 */
__global__ void warm_up_type_1 (float* spins, float* p_T, int* p_length, curandState_t* states) {
    int length = *p_length;
    float T = *p_T;
    int dim1 = blockIdx.x;
    int dim2 = blockIdx.x%2 + (threadIdx.x * 2);
    long long global_id = blockIdx.x * (2LL*blockDim.x) + blockIdx.x%2 + (threadIdx.x * 2LL);
    float upper_spin, lower_spin, left_spin, right_spin;
    float current_energy, changed_energy, delta_energy;
    float new_spin, current_spin;
/*
    // printf("%f\n", spins[16383]);
    if (global_id == 16383) 
        printf("%f", spins[(dim1-1) * length + dim2]);
*/    

    // get current position spin
    current_spin = spins[global_id];

    // get upper spin
    if (dim1 != 0) {
        upper_spin = spins[(dim1-1) * length + dim2];
    } else {
        upper_spin = spins[(length-1) * length + dim2];
    }

    // get lower spin
    if (dim1 != length-1) {
        lower_spin = spins[(dim1+1) * length + dim2];
    } else {
        lower_spin = spins[0 * length + dim2];
    }

    // get left spin
    if (dim2 != 0) {
        left_spin = spins[dim1 * length + (dim2-1)];
    } else {
        left_spin = spins[dim1 * length + (length-1)];
    }

    // get right spin
    if (dim2 != length-1) {
        right_spin = spins[dim1 * length + (dim2+1)];
    } else {
        right_spin = spins[dim1 * length + 0];
    }

    // get energy
    current_energy = -(std::cos(current_spin - upper_spin) + std::cos(current_spin - lower_spin)
                       + std::cos(current_spin - left_spin) + std::cos(current_spin - right_spin));

    // new spin direction
    new_spin = curand_uniform(&states[global_id]) * 2*M_PI;

    // changed energy
    changed_energy = -(std::cos(new_spin - upper_spin) + std::cos(new_spin - lower_spin)
                       + std::cos(new_spin - left_spin) + std::cos(new_spin - right_spin));
 
    // decide whether change spin
    delta_energy = changed_energy - current_energy;
    if ( curand_uniform(&states[global_id]) < std::exp(-delta_energy/T) ) {
        spins[global_id] = new_spin;
    }

}


/* ------------------------------------------------------------------------------------------------
 * warm up the system
 * warm up type 2 will warm up these kind of points. 1 means warm up, 0 means not warm up.
 * each 1 will be warm up one time
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   0 1 0 1 0 1 0 1 ...
 *   1 0 1 0 1 0 1 0 ...
 *   . . . . . . . . .
 *   . . . . . . . .  .
 *   . . . . . . . .    .
 */
__global__ void warm_up_type_2 (float* spins, float* p_T, int* p_length, curandState_t* states) {
    int length = *p_length;
    float T = *p_T;
    int dim1 = blockIdx.x;
    int dim2 = (1-blockIdx.x%2) + (threadIdx.x * 2);
    long long global_id = blockIdx.x * (2*blockDim.x) + (1-blockIdx.x%2) + (threadIdx.x * 2);
    float upper_spin, lower_spin, left_spin, right_spin;
    float current_energy, changed_energy, delta_energy;
    float new_spin, current_spin;

    // get current position spin
    current_spin = spins[global_id];

    // get upper spin
    if (dim1 != 0) {
        upper_spin = spins[(dim1-1) * length + dim2];
    } else {
        upper_spin = spins[(length-1) * length + dim2];
    }

    // get lower spin
    if (dim1 != length-1) {
        lower_spin = spins[(dim1+1) * length + dim2];
    } else {
        lower_spin = spins[0 * length + dim2];
    }

    // get left spin
    if (dim2 != 0) {
        left_spin = spins[dim1 * length + (dim2-1)];
    } else {
        left_spin = spins[dim1 * length + (length-1)];
    }

    // get right spin
    if (dim2 != length-1) {
        right_spin = spins[dim1 * length + (dim2+1)];
    } else {
        right_spin = spins[dim1 * length + 0];
    }

    // get energy
    current_energy = -(std::cos(current_spin - upper_spin) + std::cos(current_spin - lower_spin)
                       + std::cos(current_spin - left_spin) + std::cos(current_spin - right_spin));

    // new spin direction
    new_spin = curand_uniform(&states[global_id]) * 2*M_PI;

    // changed energy
    changed_energy = -(std::cos(new_spin - upper_spin) + std::cos(new_spin - lower_spin)
                       + std::cos(new_spin - left_spin) + std::cos(new_spin - right_spin));

    // decide whether change spin
    delta_energy = changed_energy - current_energy;
    if ( curand_uniform(&states[global_id]) < std::exp(-delta_energy/T) ) {
        spins[global_id] = new_spin;
    }
}


