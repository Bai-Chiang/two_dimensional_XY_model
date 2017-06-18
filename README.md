# two-dimensional XY model

2D-XY model simulation with CUDA
Program tested on Ubuntu 16.04 + CUDA 8.0
Run program with command:

$ nvcc -std=c++11 vortex_configuration.cu src/xy_model.cu
or
$ nvcc -std=c++11 magnetisation_susceptibility_specific_heat_vs_temperature.cu src/xy_model.cu src/cuda_reduction.cu
or 
$ nvcc -std=c++11 Tc_vs_size.cu src/xy_model.cu src/cuda_reduction.cu

You can change parameters in 
vortex_configuration.cu
magnetisation_susceptibility_specific_heat_vs_temperature.cu
Tc_vs_size.cu

output file name will change according to your parameters
output result store in result/<cuda file name>/xxx.data


Then run in command line:
$ python quiver_plot.py xxx.data
or
$ python plot.py xxx.data

Now you can check your figure xxx.eps
