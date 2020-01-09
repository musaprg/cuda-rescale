cmake_minimum_required(VERSION 3.10)

add_executable(sample "")

target_sources(sample
    PRIVATE
        examples.hpp
)

add_library(cudalib
        cuda.hpp
        functions.h
        functions.cu)

set_target_properties(cudalib PROPERTIES
        CUDA_SEPERABLE_COMPILATION ON)

target_sources(sample
        PRIVATE
        bridges.h
        bridges.cpp
        timer.hpp
        main.cpp
        )

target_link_libraries(sample cudalib)