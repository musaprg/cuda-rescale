cmake_minimum_required(VERSION 3.10)

if (NOT DEFINED CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()

project(sample LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

#set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${sample_SOURCE_DIR}/bin)

#set_property(TARGET sample PROPERTY CXX_STANDARD 11)
#set_property(TARGET sample PROPERTY CUDA_STANDARD 11)

cmake_policy(SET CMP0079 NEW)

#include(CheckLanguage)
#check_language(CUDA)
#enable_language("CUDA")
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true") # Enable separate compilation mode
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_60,code=sm_60") # P100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_61,code=sm_61") # GTX1060
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_70,code=sm_70") # V100

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(src)

find_package(SEAL 3.3.2 EXACT REQUIRED)

target_link_libraries(sample SEAL::seal)
target_link_libraries(sample ${CUDA_LIBRARIES})

target_link_libraries(bench SEAL::seal)
target_link_libraries(bench ${CUDA_LIBRARIES})