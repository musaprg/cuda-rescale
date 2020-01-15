//
// Created by kotaro on 2019/12/13.
//

#pragma once

#include "cuda.hpp"
#include "cuda_context_data.h"
//#include "seal/seal.h"
//#include "seal/context.h"

using CuCiphertext = vector<uint64_t>;
using uint64_t_array = uint64_t *;
using uint64_t_array_ptr = uint64_t_array *;
// using CuCiphertext = std::unique_ptr<uint64_t_array_ptr>;

// void initialize(std::shared_ptr<seal::SEALContext> context);

void rescale_to_next(const CuCiphertext &encrypted, CuCiphertext &destination,
                     const CudaContextData &context);

inline void rescale_to_next_inplace(CuCiphertext &encrypted,
                                    const CudaContextData &context) {
  rescale_to_next(encrypted, encrypted, context);
}

__device__ void transform_from_ntt_inplace(uint64_t_array encrypted_ntt);

__device__ void transform_to_ntt_inplace();

__device__ void transform_to_ntt_inplace(uint64_t_array encrypted);

__device__ void barret_reduce_63();

__device__ void modulo_poly_coeffs_63();

__device__ void sub_uint_uint_mod();

__device__ void sub_poly_poly_coeffmod();

__device__ void multiply_poly_scalar_coeffmod();

__global__ void mod_switch_scale_to_next(uint64_t_array encrypted,
                                         uint64_t_array destination,
                                         CudaContextData *context);

// For compiling test
__global__ void kernel(void);
void proxy(void);