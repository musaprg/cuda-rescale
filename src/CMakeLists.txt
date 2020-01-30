cmake_minimum_required(VERSION 3.10)

add_executable(sample "")

target_sources(sample
    PRIVATE
        examples.hpp
)

add_library(cudalib
        cuda.hpp
        cuda_context_data.h
        functions.h
        functions.cu)

set_target_properties(cudalib PROPERTIES
        CUDA_SEPERABLE_COMPILATION ON)
set_target_properties(cudalib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_sources(sample
        PRIVATE
        cuda_context_data.h
        bridges.h
        bridges.cpp
        timer.hpp
        main.cpp
        )

target_link_libraries(sample cudalib)