//
// Created by kotaro on 2019/12/13.
//

#pragma once

#include <thrust/transform.h>

#include "cuda.hpp"
#include "cuda_context_data.h"
//#include "seal/seal.h"
//#include "seal/context.h"

// TODO: Rewrite all functions with SFINAE (enable_if)

using namespace std;

constexpr size_t ENCRYPTED_SIZE = 2;

using CuCiphertext = vector<uint64_t>;
using int_array = int *;
using uint64_t_array = uint64_t *;
using uint64_t_array_ptr = uint64_t_array *;
using uint64_t_array_ptr_ptr = uint64_t_array_ptr *;
// using CuCiphertext = std::unique_ptr<uint64_t_array_ptr>;

// void initialize(std::shared_ptr<seal::SEALContext> context);

template <typename T>
inline void print_vector_hoge(const T &v)
{
    for (auto &&item : v)
    {
        std::cout << item << " ";
    }
    std::cout << endl;
}

__device__ inline void set_uint_uint(const uint64_t *value, size_t uint64_count,
                                     uint64_t *result)
{
    for (size_t i = 0; i < uint64_count; i++)
    {
        result[i] = value[i];
    }
}

__device__ inline void set_poly_poly(const uint64_t *poly, size_t coeff_count,
                                     size_t coeff_uint64_count,
                                     uint64_t *result)
{
    set_uint_uint(poly, coeff_count * coeff_uint64_count, result);
}
template <typename T, typename S>
__device__ [[nodiscard]] inline unsigned char add_uint64(
  T operand1, S operand2, unsigned char carry, unsigned long long *result)
{
    operand1 += operand2;
    *result = operand1 + carry;
    return (operand1 < operand2) || (~operand1 < carry);
}

template <typename T, typename S, typename R>
__device__ [[nodiscard]] inline unsigned char add_uint64(T operand1, S operand2,
                                                         R *result)
{
    *result = operand1 + operand2;
    return static_cast<unsigned char>(*result < operand1);
}

// sub_uint64_generic(operand1, operand2, borrow, result)
template <typename T, typename S>
__device__ [[nodiscard]] inline unsigned char sub_uint64(
  T operand1, S operand2, unsigned char borrow, unsigned long long *result)
{
    auto diff = operand1 - operand2;
    *result = diff - (borrow != 0);
    return (diff > operand1) || (diff < borrow);
}

template <typename T, typename S, typename R>
__device__ [[nodiscard]] inline unsigned char sub_uint64(T operand1, S operand2,
                                                         R *result)
{
    *result = operand1 - operand2;
    return static_cast<unsigned char>(operand2 > operand1);
}

__device__ inline uint64_t sub_uint_uint_mod(uint64_t operand1,
                                             uint64_t operand2,
                                             const uint64_t modulus)
{
    unsigned long long temp;
    int64_t borrow = sub_uint64(operand1, operand2, 0, &temp);
    return static_cast<uint64_t>(temp) +
           (modulus & static_cast<uint64_t>(-borrow));
};

__device__ inline void sub_poly_poly_coeffmod(const uint64_t *operand1,
                                              const uint64_t *operand2,
                                              size_t coeff_count,
                                              const uint64_t modulus,
                                              uint64_t *result)
{
    for (; coeff_count--; result++, operand1++, operand2++)
    {
        unsigned long long temp_result;
        int64_t borrow = sub_uint64(*operand1, *operand2, &temp_result);
        *result = temp_result = (modulus & static_cast<uint64_t>(-borrow));
    }
};

__device__ void multiply_poly_scalar_coeffmod(
  const uint64_t *poly, size_t coeff_count, uint64_t scalar,
  const uint64_t modulus, const uint64_t_array const_ratio, uint64_t *result);

// multiply_uint64_generic
template <typename T, typename S>
__device__ inline void multiply_uint64(T operand1, S operand2,
                                       unsigned long long *result128)
{
    auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
    auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
    operand1 >>= 32;
    operand2 >>= 32;

    auto middle1 = operand1 * operand2_coeff_right;
    uint64_t middle;
    auto left = operand1 * operand2 +
                (static_cast<uint64_t>(add_uint64(
                   middle1, operand2 * operand1_coeff_right, &middle))
                 << 32);
    auto right = operand1_coeff_right * operand2_coeff_right;
    auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

    result128[1] =
      static_cast<unsigned long long>(left + (middle >> 32) + (temp_sum >> 32));
    result128[0] = static_cast<unsigned long long>(
      (temp_sum << 32) | (right & 0x00000000FFFFFFFF));
}

// multiply_uint64_hw64_generic
__device__ inline void multiply_uint64_hw64(uint64_t operand1,
                                            uint64_t operand2,
                                            unsigned long long *hw64)
{
    auto operand1_coeff_right = operand1 & 0x00000000FFFFFFFF;
    auto operand2_coeff_right = operand2 & 0x00000000FFFFFFFF;
    operand1 >>= 32;
    operand2 >>= 32;

    auto middle1 = operand1 * operand2_coeff_right;
    uint64_t middle;
    auto left = operand1 * operand2 +
                (static_cast<uint64_t>(add_uint64(
                   middle1, operand2 * operand1_coeff_right, &middle))
                 << 32);
    auto right = operand1_coeff_right * operand2_coeff_right;
    auto temp_sum = (right >> 32) + (middle & 0x00000000FFFFFFFF);

    *hw64 =
      static_cast<unsigned long long>(left + (middle >> 32) + (temp_sum >> 32));
}

void rescale_to_next(const CuCiphertext &encrypted, CuCiphertext &destination,
                     const CudaContextData &context);

inline void rescale_to_next_inplace(CuCiphertext &encrypted,
                                    const CudaContextData &context)
{
    rescale_to_next(encrypted, encrypted, context);
}

__device__ inline uint64_t barret_reduce_63(
  uint64_t input, uint64_t modulus,
  const uint64_t_array const_ratio // SmallModulus::const_ratio_.data()
)
{
    // Reduces input using base 2^64 Barrett reduction
    // input must be at most 63 bits

    unsigned long long tmp[2];
    multiply_uint64(input, const_ratio[1], tmp);

    // Barrett subtraction
    tmp[0] = input - tmp[1] * modulus;

    // One more subtraction is enough
    return static_cast<std::uint64_t>(tmp[0]) -
           (modulus & static_cast<std::uint64_t>(
                        -static_cast<std::int64_t>(tmp[0] >= modulus)));
}

__device__ inline void modulo_poly_coeffs_63(const uint64_t_array poly,
                                             size_t coeff_count,
                                             const uint64_t modulus,
                                             const uint64_t_array const_ratio,
                                             uint64_t_array result)
{
    // TODO: If this does not work, remove it.
    thrust::transform(poly, poly + coeff_count, result, [&](uint64_t coeff) {
        return barret_reduce_63(coeff, modulus, const_ratio);
    });
}

// Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and
// Michael Naehrig).
__device__ void inverse_ntt_negacyclic_harvey_lazy(
  uint64_t_array operand, uint64_t modulus, int coeff_count_power,
  uint64_t_array inv_root_powers_div_two,
  uint64_t_array scaled_inv_root_powers_div_two);

__device__ inline void inverse_ntt_negacyclic_harvey(
  uint64_t_array operand, int coeff_count_power, uint64_t modulus,
  uint64_t_array inv_root_powers_div_two,
  uint64_t_array scaled_inv_root_powers_div_two)
{
    inverse_ntt_negacyclic_harvey_lazy(operand, modulus, coeff_count_power,
                                       inv_root_powers_div_two,
                                       scaled_inv_root_powers_div_two);

    // Finally maybe we need to reduce every coefficient modulo q, but we
    // know that they are in the range [0, 4q).
    // Since word size is controlled this is fast.
    uint64_t two_times_modulus = modulus * 2;
    size_t n = size_t(1) << coeff_count_power;

    for (; n--; operand++)
    {
        if (*operand >= two_times_modulus)
        {
            *operand -= two_times_modulus;
        }
        if (*operand >= modulus)
        {
            *operand -= modulus;
        }
    }
}

/**
This function computes in-place the negacyclic NTT. The input is
a polynomial a of degree n in R_q, where n is assumed to be a power of
2 and q is a prime such that q = 1 (mod 2n).

The output is a vector A such that the following hold:
A[j] =  a(psi**(2*bit_reverse(j) + 1)), 0 <= j < n.

For details, see Michael Naehrig and Patrick Longa.
*/
__device__ void ntt_negacyclic_harvey_lazy(uint64_t_array operand,
                                           uint64_t modulus,
                                           int coeff_count_power,
                                           uint64_t_array root_powers,
                                           uint64_t_array scaled_root_powers);

__device__ inline void ntt_negacyclic_harvey(uint64_t_array operand,
                                             int coeff_count_power,
                                             uint64_t modulus,
                                             uint64_t_array root_powers,
                                             uint64_t_array scaled_root_powers)
{
    ntt_negacyclic_harvey_lazy(operand, modulus, coeff_count_power, root_powers,
                               scaled_root_powers);

    size_t n = size_t(1) << coeff_count_power;

    // Final adjustments; compute a[j] = a[j] * n^{-1} mod q.
    // We incorporated the final adjustment in the butterfly. Only need
    // to reduce here.
    for (; n--; operand++)
    {
        if (*operand >= modulus)
        {
            *operand -= modulus;
        }
    }
}

__device__ void transform_from_ntt_inplace(
  uint64_t_array encrypted_ntt, // ciphertext
  uint64_t_array coeff_modulus,
  size_t coeff_modulus_count, // coeff modulus
  size_t coeff_count,         // poly_modulus_degree
  int coeff_count_power,      // lg(poly_modulus_degree)
  uint64_t_array_ptr ntt_inv_root_powers_div_two,
  uint64_t_array_ptr ntt_scaled_inv_root_powers_div_two);

__device__ void transform_to_ntt_inplace(
  uint64_t_array encrypted, // ciphertext
  uint64_t_array coeff_modulus,
  size_t coeff_modulus_count, // coeff modulus
  size_t coeff_count,         // poly_modulus_degree
  int coeff_count_power,      // lg(poly_modulus_degree)
  uint64_t_array_ptr ntt_root_powers,
  uint64_t_array_ptr ntt_scaled_root_powers);

__global__ void mod_switch_scale_to_next(
  const uint64_t_array encrypted, uint64_t_array destination,
  const uint64_t_array coeff_modulus, const uint64_t_array next_coeff_modulus,
  size_t encrypted_size, size_t destination_size, size_t coeff_modulus_count,
  size_t next_coeff_modulus_size, size_t coeff_count, int coeff_count_power,
  uint64_t_array_ptr ntt_root_powers, uint64_t_array_ptr ntt_scaled_root_powers,
  uint64_t_array_ptr ntt_inv_root_powers_div_two,
  uint64_t_array_ptr ntt_scaled_inv_root_powers_div_two);

// For compiling test
__global__ void kernel(void);
void proxy(void);