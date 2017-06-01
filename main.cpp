//
// Created by Baiqiang Qiang on 30/05/2017.
//

#include <iostream>
#include <cmath>

int main() {
    std::cout << "hello world" << std::endl;
    long long n_sample = 100;
    int size = 100;

    // ========================================
    float** pp_spins = new float* [n_sample];
    for (long long i = 0; i < n_sample; ++i) {
        pp_spins[i] = new float [size*size];
    }

    // =======================================
    // free pp_spins
    for (long long i = 0; i < n_sample; ++i) {
        delete[] pp_spins[i];
    }
    delete[] pp_spins;
}
