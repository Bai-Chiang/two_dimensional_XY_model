# ifndef XY_MODEL_H_
# define XY_MODEL_H_

__global__ void init_rand(unsigned int seed, curandState_t* states);
__global__ void initialize (float* spins, curandState_t* states);
__global__ void warm_up (float* spins, float* p_T, int* p_length, long long* p_n_itter, curandState_t* states);
__global__ void get_spin(float* Sx, float* Sy, float* theta);
__global__ void get_energy (float* energy, float* spins, int* p_length);
__global__ void warm_up_type_1 (float* spins, float* p_T, int* p_length, curandState_t* states);
__global__ void warm_up_type_2 (float* spins, float* p_T, int* p_length, curandState_t* states);
# endif
