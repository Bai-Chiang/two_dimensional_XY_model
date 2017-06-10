cmake_minimum_required(VERSION 3.6)   # CMake version check
project(two_dimensional_XY_model)     # Create project
set(CMAKE_CXX_STANDARD 11)            # Enable c++11 standard

set(SOURCE_FILES main.cpp)            # Add main.cpp file of project root directory as source file
add_executable(2d_xy.out ${SOURCE_FILES})       # Add executable target with source files listed in SOURCE_FILES variable