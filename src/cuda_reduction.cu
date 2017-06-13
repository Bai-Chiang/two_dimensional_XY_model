// --- file: cuda_reduction.cu ---
// reduce float array

#include <iostream>
#include "cuda_reduction.h"

/* ------------------------------------------------------------------------------------------------
 * reduce array using gpu
 */
__global__ void cuda_reduce (float* iarray, float* oarray) {
    extern __shared__ float sarray[];
    long long global_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    
    sarray[threadIdx.x] = iarray[global_id] + iarray[global_id + blockDim.x];
    __syncthreads();

    if (blockDim.x >= 512) {
        if (threadIdx.x < 256) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 256];
            __syncthreads();
        }
    }
    if (blockDim.x >= 256) {
        if (threadIdx.x < 128) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 128];
            __syncthreads();
        }
    }
    if (blockDim.x >= 128) {
        if (threadIdx.x < 64) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 64];
            __syncthreads();
        }
    }


    if (threadIdx.x < 32){
        if (blockDim.x >= 64) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 32];
            __syncthreads();
        }
        if (blockDim.x >= 32) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 16];
            __syncthreads();
        }
        if (blockDim.x >= 16) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 8];
            __syncthreads();
        }
        if (blockDim.x >= 8) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 4];
            __syncthreads();
        }
        if (blockDim.x >= 4) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 2];
            __syncthreads();
        }
        if (blockDim.x >= 2) {
            sarray[threadIdx.x] += sarray[threadIdx.x + 1];
            __syncthreads();
        }
    }

    if (threadIdx.x == 0) {
        oarray[blockIdx.x] = sarray[0];
    }

}

