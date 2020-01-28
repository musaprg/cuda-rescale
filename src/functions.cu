//
// Created by kotaro on 2019/12/13.
//

#include "functions.h"
#include <cstdio>
#include <iostream>

/*
 * 1. pour data from Ciphertext
 * 2. memcpy to GPU
 * 3. perform evaluation
 * 4. send back to host
 * 5. Update Ciphertext instances
 */

/*
 * NOTE:
 * small_ntt_tables contains 'coeff_mod_count' of small_ntt_table
 */

// ---------------------------------------------------------------------------

void rescale_to_next(const CuCiphertext &encrypted, CuCiphertext &destination,
                     const CudaContextData &context) {
  // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA%A4%C7%B9%D4%CE%F3%B1%E9%BB%BB%A1%A7%B2%C3%B8%BA%BB%BB

  size_t encrypted_size = encrypted.size();
  // TODO: precaliculate destination size from input parameters.
  size_t destination_size = destination.size();
  size_t coeff_count = context.coeff_count;
  size_t coeff_modulus_size = context.coeff_modulus.size();
  size_t next_coeff_modulus_size = context.next_coeff_modulus.size();
  size_t next_ciphertext_size =
      coeff_count * ENCRYPTED_SIZE * next_coeff_modulus_size;

  auto device_encrypted = cuda::make_unique<uint64_t[]>(encrypted_size);
  auto device_destination = cuda::make_unique<uint64_t[]>(destination_size);
  auto device_coeff_modulus = cuda::make_unique<uint64_t[]>(coeff_modulus_size);
  auto device_next_coeff_modulus =
      cuda::make_unique<uint64_t[]>(next_coeff_modulus_size);
  auto device_temp1 = cuda::make_unique<uint64_t[]>(coeff_count);          // t
  auto device_temp2 = cuda::make_unique<uint64_t[]>(next_ciphertext_size); // u

  cuda::CHECK_CUDA_ERROR(::cudaMemcpy(device_encrypted.get(), encrypted.data(),
                                      sizeof(uint64_t) * encrypted_size,
                                      cudaMemcpyHostToDevice));
  cuda::CHECK_CUDA_ERROR(::cudaMemcpy(
      device_coeff_modulus.get(), context.coeff_modulus.data(),
      sizeof(uint64_t) * coeff_modulus_size, cudaMemcpyHostToDevice));
  cuda::CHECK_CUDA_ERROR(::cudaMemcpy(
      device_next_coeff_modulus.get(), context.next_coeff_modulus.data(),
      sizeof(uint64_t) * next_coeff_modulus_size, cudaMemcpyHostToDevice));
  //  ::cudaDeviceSynchronize(); // synchronize

  /*
   * If the device memory leaks, expand heap size with this (default 8MB)
   * This expand heap size up to 1GB
   * http://dfukunaga.hatenablog.com/entry/2017/10/28/163538
   */
  //  size_t device_heap_size = 1024 * 1024 * 1024;
  //  cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);

  int thread_per_block = 256;
  int num_block = (encrypted_size + thread_per_block - 1) / thread_per_block;
  //  mod_switch_scale_to_next<<<num_block, thread_per_block>>>(
  //      device_encrypted.get(), device_destination.get(),
  //      device_coeff_modulus.get(), device_next_coeff_modulus.get(),
  //      encrypted_size, destination_size, coeff_modulus_size,
  //      next_coeff_modulus_size, coeff_count);
  auto cudaStatus = ::cudaDeviceSynchronize(); // synchronize

  if (cudaStatus != ::cudaSuccess) {
    throw logic_error("cudaDeviceSynchronize returned error code");
  }

  cuda::CHECK_CUDA_ERROR(::cudaMemcpy(
      destination.data(), device_destination.get(),
      sizeof(uint64_t) * destination_size, cudaMemcpyDeviceToHost));

  print_vector_hoge(encrypted);
  print_vector_hoge(destination);
}

// TODO: Fix header to suit this definition
// TODO: Implement me!!!!!!!!!!!!!!!!!!!!!!!
__global__ void mod_switch_scale_to_next(
    const uint64_t_array encrypted, uint64_t_array destination,
    const uint64_t_array coeff_modulus, const uint64_t_array next_coeff_modulus,
    size_t encrypted_size, size_t destination_size, size_t coeff_modulus_count,
    size_t next_coeff_modulus_size, size_t coeff_count, int coeff_count_power,
    uint64_t_array_ptr ntt_root_powers,
    uint64_t_array_ptr ntt_scaled_root_powers,
    uint64_t_array_ptr ntt_inv_root_powers_div_two,
    uint64_t_array_ptr ntt_scaled_inv_root_powers_div_two) {
  //  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  transform_from_ntt_inplace(encrypted, coeff_modulus, coeff_modulus_count,
                             coeff_count, coeff_count_power,
                             ntt_inv_root_powers_div_two,
                             ntt_scaled_inv_root_powers_div_two);

  for (size_t poly_index = 0; poly_index < ENCRYPTED_SIZE; poly_index++) {
  }

  transform_to_ntt_inplace(encrypted, coeff_modulus, coeff_modulus_count,
                           coeff_count, coeff_count_power, ntt_root_powers,
                           ntt_scaled_root_powers);
}

__device__ void multiply_poly_scalar_coeffmod(const uint64_t *poly,
                                              size_t coeff_count,
                                              uint64_t scalar,
                                              const uint64_t modulus,
                                              const uint64_t_array const_ratio,
                                              uint64_t *result) {
  // Explicit inline
  // for (int i = 0; i < coeff_count; i++)
  //{
  //    *result++ = multiply_uint_uint_mod(*poly++, scalar, modulus);
  //}
  const uint64_t const_ratio_0 = const_ratio[0];
  const uint64_t const_ratio_1 = const_ratio[1];
  for (; coeff_count--; poly++, result++) {
    unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
    multiply_uint64(*poly, scalar, z);

    // Reduces z using base 2^64 Barrett reduction

    // Multiply input and const_ratio
    // Round 1
    multiply_uint64_hw64(z[0], const_ratio_0, &carry);
    multiply_uint64(z[0], const_ratio_1, tmp2);
    tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

    // Round 2
    multiply_uint64(z[1], const_ratio_0, tmp2);
    carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

    // This is all we care about
    tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

    // Barrett subtraction
    tmp3 = z[0] - tmp1 * modulus;

    // Claim: One more subtraction is enough
    *result = tmp3 - (modulus & static_cast<uint64_t>(
                                    -static_cast<int64_t>(tmp3 >= modulus)));
  }
}

// TODO: change to default cuda kernel(__global__)
__device__ void transform_from_ntt_inplace(
    uint64_t_array encrypted_ntt, // ciphertext
    uint64_t_array coeff_modulus,
    size_t coeff_modulus_count, // coeff modulus
    size_t coeff_count,         // poly_modulus_degree
    int coeff_count_power,      // lg(poly_modulus_degree)
    uint64_t_array_ptr ntt_inv_root_powers_div_two,
    uint64_t_array_ptr ntt_scaled_inv_root_powers_div_two) {
  for (size_t i = 0; i < ENCRYPTED_SIZE; i++) {
    for (size_t j = 0; j < coeff_modulus_count; j++) {
      inverse_ntt_negacyclic_harvey(encrypted_ntt + i + j * coeff_count,
                                    coeff_count_power, coeff_modulus[j],
                                    ntt_inv_root_powers_div_two[j],
                                    ntt_scaled_inv_root_powers_div_two[j]);
    }
  }
}

// TODO: change to default cuda kernel(__global__)
__device__ void
transform_to_ntt_inplace(uint64_t_array encrypted, uint64_t_array coeff_modulus,
                         size_t coeff_modulus_count, size_t coeff_count,
                         int coeff_count_power,
                         uint64_t_array_ptr ntt_root_powers,
                         uint64_t_array_ptr ntt_scaled_root_powers) {
  for (size_t i = 0; i < ENCRYPTED_SIZE; i++) {
    for (size_t j = 0; j < coeff_modulus_count; j++) {
      ntt_negacyclic_harvey(encrypted + i + j * coeff_count, coeff_count_power,
                            coeff_modulus[j], ntt_root_powers[j],
                            ntt_scaled_root_powers[j]);
    }
  }
}

// Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and
// Michael Naehrig).
__device__ void inverse_ntt_negacyclic_harvey_lazy(
    uint64_t_array operand, uint64_t modulus, int coeff_count_power,
    uint64_t_array inv_root_powers_div_two,
    uint64_t_array scaled_inv_root_powers_div_two) {
  uint64_t two_times_modulus = modulus * 2;

  // return the bit-reversed order of NTT.
  size_t n = size_t(1) << coeff_count_power;
  size_t t = 1;

  for (size_t m = n; m > 1; m >>= 1) {
    size_t j1 = 0;
    size_t h = m >> 1;
    if (t >= 4) {
      for (size_t i = 0; i < h; i++) {
        size_t j2 = j1 + t;
        // Need the powers of phi^{-1} in bit-reversed order
        const uint64_t W = inv_root_powers_div_two[h + i];
        const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];

        uint64_t *U = operand + j1;
        uint64_t *V = U + t;
        uint64_t currU;
        uint64_t T;
        unsigned long long H;
        for (size_t j = j1; j < j2; j += 4) {
          T = two_times_modulus - *V + *U;
          currU =
              *U + *V -
              (two_times_modulus &
               static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
          *U++ = (currU + (modulus & static_cast<uint64_t>(
                                         -static_cast<int64_t>(T & 1)))) >>
                 1;
          multiply_uint64_hw64(Wprime, T, &H);
          *V++ = T * W - H * modulus;

          T = two_times_modulus - *V + *U;
          currU =
              *U + *V -
              (two_times_modulus &
               static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
          *U++ = (currU + (modulus & static_cast<uint64_t>(
                                         -static_cast<int64_t>(T & 1)))) >>
                 1;
          multiply_uint64_hw64(Wprime, T, &H);
          *V++ = T * W - H * modulus;

          T = two_times_modulus - *V + *U;
          currU =
              *U + *V -
              (two_times_modulus &
               static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
          *U++ = (currU + (modulus & static_cast<uint64_t>(
                                         -static_cast<int64_t>(T & 1)))) >>
                 1;
          multiply_uint64_hw64(Wprime, T, &H);
          *V++ = T * W - H * modulus;

          T = two_times_modulus - *V + *U;
          currU =
              *U + *V -
              (two_times_modulus &
               static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));
          *U++ = (currU + (modulus & static_cast<uint64_t>(
                                         -static_cast<int64_t>(T & 1)))) >>
                 1;
          multiply_uint64_hw64(Wprime, T, &H);
          *V++ = T * W - H * modulus;
        }
        j1 += (t << 1);
      }
    } else {
      for (size_t i = 0; i < h; i++) {
        size_t j2 = j1 + t;
        // Need the powers of  phi^{-1} in bit-reversed order
        const uint64_t W = inv_root_powers_div_two[h + i];
        const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];

        uint64_t *U = operand + j1;
        uint64_t *V = U + t;
        uint64_t currU;
        uint64_t T;
        unsigned long long H;
        for (size_t j = j1; j < j2; j++) {
          // U = x[i], V = x[i+m]

          // Compute U - V + 2q
          T = two_times_modulus - *V + *U;

          // Cleverly check whether currU + currV >= two_times_modulus
          currU =
              *U + *V -
              (two_times_modulus &
               static_cast<uint64_t>(-static_cast<int64_t>((*U << 1) >= T)));

          // Need to make it so that div2_uint_mod takes values that are > q.
          // div2_uint_mod(U, modulusptr, coeff_uint64_count, U);
          // We use also the fact that parity of currU is same as parity of T.
          // Since our modulus is always so small that currU + masked_modulus <
          // 2^64, we never need to worry about wrapping around when adding
          // masked_modulus.
          // uint64_t masked_modulus = modulus &
          // static_cast<uint64_t>(-static_cast<int64_t>(T & 1)); uint64_t carry
          // = add_uint64(currU, masked_modulus, 0, &currU); currU += modulus &
          // static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
          *U++ = (currU + (modulus & static_cast<uint64_t>(
                                         -static_cast<int64_t>(T & 1)))) >>
                 1;

          multiply_uint64_hw64(Wprime, T, &H);
          // effectively, the next two multiply perform multiply modulo beta =
          // 2**wordsize.
          *V++ = W * T - H * modulus;
        }
        j1 += (t << 1);
      }
    }
    t <<= 1;
  }
}

__device__ void ntt_negacyclic_harvey_lazy(uint64_t_array operand,
                                           uint64_t modulus,
                                           int coeff_count_power,
                                           uint64_t_array root_powers,
                                           uint64_t_array scaled_root_powers) {
  auto two_times_modulus = modulus * 2;

  // Return the NTT in scrambled order
  size_t n = size_t(1) << coeff_count_power;
  size_t t = n >> 1;
  for (size_t m = 1; m < n; m <<= 1) {
    if (t >= 4) {
      for (size_t i = 0; i < m; i++) {
        size_t j1 = 2 * i * t;
        size_t j2 = j1 + t;
        const uint64_t W = root_powers[m + i];
        const uint64_t Wprime = scaled_root_powers[m + i];

        uint64_t *X = operand + j1;
        uint64_t *Y = X + t;
        uint64_t currX;
        unsigned long long Q;
        for (size_t j = j1; j < j2; j += 4) {
          currX = *X - (two_times_modulus &
                        static_cast<uint64_t>(
                            -static_cast<int64_t>(*X >= two_times_modulus)));
          multiply_uint64_hw64(Wprime, *Y, &Q);
          Q = *Y * W - Q * modulus;
          *X++ = currX + Q;
          *Y++ = currX + (two_times_modulus - Q);

          currX = *X - (two_times_modulus &
                        static_cast<uint64_t>(
                            -static_cast<int64_t>(*X >= two_times_modulus)));
          multiply_uint64_hw64(Wprime, *Y, &Q);
          Q = *Y * W - Q * modulus;
          *X++ = currX + Q;
          *Y++ = currX + (two_times_modulus - Q);

          currX = *X - (two_times_modulus &
                        static_cast<uint64_t>(
                            -static_cast<int64_t>(*X >= two_times_modulus)));
          multiply_uint64_hw64(Wprime, *Y, &Q);
          Q = *Y * W - Q * modulus;
          *X++ = currX + Q;
          *Y++ = currX + (two_times_modulus - Q);

          currX = *X - (two_times_modulus &
                        static_cast<uint64_t>(
                            -static_cast<int64_t>(*X >= two_times_modulus)));
          multiply_uint64_hw64(Wprime, *Y, &Q);
          Q = *Y * W - Q * modulus;
          *X++ = currX + Q;
          *Y++ = currX + (two_times_modulus - Q);
        }
      }
    } else {
      for (size_t i = 0; i < m; i++) {
        size_t j1 = 2 * i * t;
        size_t j2 = j1 + t;
        const uint64_t W = root_powers[m + i];
        const uint64_t Wprime = scaled_root_powers[m + i];

        uint64_t *X = operand + j1;
        uint64_t *Y = X + t;
        uint64_t currX;
        unsigned long long Q;
        for (size_t j = j1; j < j2; j++) {
          // The Harvey butterfly: assume X, Y in [0, 2p), and return X', Y' in
          // [0, 2p). X', Y' = X + WY, X - WY (mod p).
          currX = *X - (two_times_modulus &
                        static_cast<uint64_t>(
                            -static_cast<int64_t>(*X >= two_times_modulus)));
          multiply_uint64_hw64(Wprime, *Y, &Q);
          Q = W * *Y - Q * modulus;
          *X++ = currX + Q;
          *Y++ = currX + (two_times_modulus - Q);
        }
      }
    }
    t >>= 1;
  }
}

__global__ void kernel(void) { printf("Hello from CUDA kernel.\n"); }

void proxy(void) { kernel<<<1, 1>>>(); }
