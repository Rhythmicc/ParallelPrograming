cmake_minimum_required(VERSION 3.17)
project(ParallelProgram C)

#include_directories("/usr/local/opt/libomp/include")
#link_directories("/usr/local/opt/libomp/lib")
include_directories("include")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-fopenmp")

add_definitions(-DLIMIT_THREAD)
add_executable(ParallelProgram main.c include)
add_executable(sdnn finalproject-SDNN/main.c)
add_executable(cholesky cholesky/main.c)