//
// Created by kotaro on 2019/12/13.
//

#pragma once

#include "cuda.hpp"

__global__ void kernel(void);

__device__ void barret_reduce_63();

__device__ void modulo_poly_coeffs_63();

__device__ void sub_uint_uint_mod();

__device__ void sub_poly_poly_coeffmod();

void proxy(void);