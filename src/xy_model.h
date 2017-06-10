# ifndef XY_MODEL_H_
# define XY_MODEL_H_

__global__ void init_rand(unsigned int seed, curandState_t* states);
__global__ void initialize (float* spins, curandState_t* states);
__global__ void warm_up (float* spins, float* p_T, int* p_length, long long* p_n_itter, curandState_t* states);

# endif
