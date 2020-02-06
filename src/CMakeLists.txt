cmake_minimum_required(VERSION 3.10)

add_executable(sample "")
add_executable(bench "")

add_library(cudalib
        cuda.hpp
        cuda_context_data.h
        timer.hpp
        functions.h
        functions.cu)

set_target_properties(cudalib PROPERTIES
        CUDA_SEPERABLE_COMPILATION ON)
set_target_properties(cudalib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_sources(sample
        PRIVATE
        cuda_context_data.h
        examples.hpp
        bridges.h
        bridges.cpp
        timer.hpp
        main.cpp
        )

target_link_libraries(sample cudalib)

target_sources(bench
        PRIVATE
        examples.hpp
        cuda_context_data.h
        bridges.h
        bridges.cpp
        timer.hpp
        performance_evaluator.h
        performance_evaluator.cpp
        )

target_link_libraries(bench cudalib)