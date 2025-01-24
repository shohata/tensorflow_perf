cmake_minimum_required(VERSION 3.5)
project(benchmark)

find_package(TensorFlow REQUIRED)
include_directories(${TensorFlow_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} benchmark.cpp)
target_link_libraries(${PROJECT_NAME} ${TensorFlow_LIBRARIES})

add_executable(benchmark_tfrecord benchmark_tfrecord.cpp)
target_link_libraries(benchmark_tfrecord ${TensorFlow_LIBRARIES})
