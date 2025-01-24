cmake_minimum_required (VERSION 3.5)

project(gpu_perf)

set(CMAKE_C_COMPILE clang)
set(CMAKE_CXX_COMPILE clang++)

add_subdirectory (src)
