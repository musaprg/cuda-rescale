//
// Created by kotaro on 2019/12/13.
//

#include <cstdio>
#include <iostream>

#include "functions.h"

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
                     const CudaContextData &context)
{
    // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA%A4%C7%B9%D4%CE%F3%B1%E9%BB%BB%A1%A7%B2%C3%B8%BA%BB%BB
    cudaSetDevice(CUDA_DEVICE_ID);

    size_t coeff_count = context.coeff_count;
    int coeff_count_power = context.coeff_count_power;
    cout << "Coeff Count: " << coeff_count << endl;
    cout << "Coeff Count Power: " << coeff_count_power << endl;
    size_t coeff_modulus_size = context.coeff_modulus.size();
    size_t coeff_modulus_const_ratio_size =
      context.coeff_modulus_const_ratio.size();
    size_t next_coeff_modulus_size = context.next_coeff_modulus.size();
    size_t next_coeff_modulus_const_ratio_size =
      context.next_coeff_modulus_const_ratio.size();
    size_t next_ciphertext_size =
      coeff_count * ENCRYPTED_SIZE * next_coeff_modulus_size;
    // TODO: 各q_lごとに存在する気がするので修正する
    // ひとまずは動かすこと優先なのでencryptedが対応するレベルのものだけ取り込む．
    size_t inv_last_coeff_mod_array_size =
      context.inv_last_coeff_mod_array.size();

    size_t encrypted_size = encrypted.size();
    // TODO: precaliculate destination size from input parameters.
    size_t destination_size =
      ENCRYPTED_SIZE * coeff_count * next_coeff_modulus_size;

    print_log("Allocate Device Memeory");
    auto device_encrypted = cuda::make_unique<uint64_t[]>(encrypted_size);
    auto device_destination = cuda::make_unique<uint64_t[]>(destination_size);
    auto device_coeff_modulus =
      cuda::make_unique<uint64_t[]>(coeff_modulus_size);
    auto device_coeff_modulus_const_ratio =
      cuda::make_unique<uint64_t[]>(coeff_modulus_const_ratio_size);
    // TODO: this may be unnecessary.
    auto device_next_coeff_modulus =
      cuda::make_unique<uint64_t[]>(next_coeff_modulus_size);
    auto device_temp1 = cuda::make_unique<uint64_t[]>(coeff_count); // t
    auto device_temp2 =
      cuda::make_unique<uint64_t[]>(next_ciphertext_size); // u
    auto device_ntt_root_powers =
      cuda::make_unique<uint64_t[]>(coeff_count * coeff_modulus_size);
    auto device_ntt_inv_root_powers_div_two =
      cuda::make_unique<uint64_t[]>(coeff_count * coeff_modulus_size);
    auto device_ntt_scaled_root_powers =
      cuda::make_unique<uint64_t[]>(coeff_count * coeff_modulus_size);
    auto device_ntt_scaled_inv_root_powers_div_two =
      cuda::make_unique<uint64_t[]>(coeff_count * coeff_modulus_size);
    auto device_inv_last_coeff_mod_array =
      cuda::make_unique<uint64_t[]>(inv_last_coeff_mod_array_size);

    print_log("Copy to Device Memeory");
    cuda::CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      device_encrypted.get(), encrypted.data(),
      sizeof(uint64_t) * encrypted_size, cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      device_coeff_modulus.get(), context.coeff_modulus.data(),
      sizeof(uint64_t) * coeff_modulus_size, cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(device_coeff_modulus_const_ratio.get(),
                        context.coeff_modulus_const_ratio.data(),
                        sizeof(uint64_t) * coeff_modulus_const_ratio_size,
                        cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      device_next_coeff_modulus.get(), context.next_coeff_modulus.data(),
      sizeof(uint64_t) * next_coeff_modulus_size, cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      device_ntt_root_powers.get(), context.ntt_root_powers.data(),
      sizeof(uint64_t) * coeff_modulus_size * coeff_count,
      cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(device_ntt_inv_root_powers_div_two.get(),
                        context.ntt_inv_root_powers_div_two.data(),
                        sizeof(uint64_t) * coeff_modulus_size * coeff_count,
                        cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(device_ntt_scaled_root_powers.get(),
                        context.ntt_scaled_root_powers.data(),
                        sizeof(uint64_t) * coeff_modulus_size * coeff_count,
                        cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(device_ntt_scaled_inv_root_powers_div_two.get(),
                        context.ntt_scaled_inv_root_powers_div_two.data(),
                        sizeof(uint64_t) * coeff_modulus_size * coeff_count,
                        cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(
      ::cudaMemcpyAsync(device_inv_last_coeff_mod_array.get(),
                        context.inv_last_coeff_mod_array.data(),
                        sizeof(uint64_t) * inv_last_coeff_mod_array_size,
                        cudaMemcpyHostToDevice));
    cuda::CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    /*
     * If the device memory leaks, expand heap size with this (default 8MB)
     * This expand heap size up to 1GB
     * http://dfukunaga.hatenablog.com/entry/2017/10/28/163538
     */
    //  size_t device_heap_size = 1024 * 1024 * 1024;
    //  cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);

    print_poly(encrypted, coeff_count, 10);

    size_t num_blocks =
      (coeff_modulus_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    print_log("Perform iNTT and Check Result");
    transform_from_ntt_inplace<<<num_blocks, THREADS_PER_BLOCK>>>(
      device_encrypted.get(), device_coeff_modulus.get(), coeff_modulus_size,
      coeff_count, coeff_count_power, device_ntt_inv_root_powers_div_two.get(),
      device_ntt_scaled_inv_root_powers_div_two.get());
    cuda::CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cuda::CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      destination.data(), device_encrypted.get(),
      sizeof(uint64_t) * encrypted_size, cudaMemcpyDeviceToHost));
    cuda::CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    print_poly(destination, coeff_count, 10);

    // print_log("Perform mod_switch_scale_to_next");
    // mod_switch_scale_to_next<<<num_blocks, THREADS_PER_BLOCK>>>(
    //   device_encrypted.get(), device_destination.get(),
    //   device_coeff_modulus.get(), device_coeff_modulus_const_ratio.get(),
    //   device_next_coeff_modulus.get(), encrypted_size, destination_size,
    //   coeff_modulus_size, next_coeff_modulus_size, coeff_count,
    //   coeff_count_power, device_ntt_root_powers.get(),
    //   device_ntt_scaled_root_powers.get(),
    //   device_ntt_inv_root_powers_div_two.get(),
    //   device_ntt_scaled_inv_root_powers_div_two.get(), device_temp1.get(),
    //   device_temp2.get(), device_inv_last_coeff_mod_array.get());
    // // assert(equal(encrypted.begin(), encrypted.end(),
    // destination.begin())); cuda::CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // transform_to_ntt_inplace<<<num_blocks, THREADS_PER_BLOCK>>>(
    //   device_encrypted.get(), device_coeff_modulus.get(), coeff_modulus_size,
    //   coeff_count, coeff_count_power, device_ntt_root_powers.get(),
    //   device_ntt_scaled_root_powers.get());
    // if (::cudaDeviceSynchronize() != ::cudaSuccess)
    // {
    //     throw logic_error("cudaDeviceSynchronize returned error code");
    // }
    // cuda::CHECK_CUDA_ERROR(::cudaMemcpy(
    //   destination.data(), device_encrypted.get(),
    //   sizeof(uint64_t) * destination_size, cudaMemcpyDeviceToHost));
    // print_vector_hoge(destination); // check result

    print_log("Get the result from GPU");
    destination.resize(destination_size);
    cuda::CHECK_CUDA_ERROR(::cudaMemcpyAsync(
      destination.data(), device_destination.get(),
      sizeof(uint64_t) * destination_size, cudaMemcpyDeviceToHost));
    cuda::CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    print_poly(destination, coeff_count, 10);
}

// TODO: Fix header to suit this definition
// TODO: Implement me!!!!!!!!!!!!!!!!!!!!!!!
__global__ void mod_switch_scale_to_next(
  uint64_t_array encrypted, uint64_t_array destination,
  const uint64_t_array coeff_modulus,
  const uint64_t_array coeff_modulus_const_ratio, // std::array<uint64_t, 3>
                                                  // SmallModulus::const_ratio_
  const uint64_t_array next_coeff_modulus, size_t encrypted_size,
  size_t destination_size, size_t coeff_modulus_size,
  size_t next_coeff_modulus_size, size_t coeff_count, int coeff_count_power,
  uint64_t_array ntt_root_powers, uint64_t_array ntt_scaled_root_powers,
  uint64_t_array ntt_inv_root_powers_div_two,
  uint64_t_array ntt_scaled_inv_root_powers_div_two,
  uint64_t_array temp1,                   // t
  uint64_t_array temp2,                   // u
  uint64_t_array inv_last_coeff_mod_array // q_l^-1 mod q_i
)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;

    size_t num_blocks =
      (coeff_modulus_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    if (tid == 0)
    {
        // NOTE: no affect...Why?
        // transform_from_ntt_inplace<<<num_blocks, THREADS_PER_BLOCK>>>(
        //   encrypted, coeff_modulus, coeff_modulus_size, coeff_count,
        //   coeff_count_power, ntt_inv_root_powers_div_two,
        //   ntt_scaled_inv_root_powers_div_two);
        // cudaDeviceSynchronize();

        auto temp2_ptr = temp2;

        // #pragma unroll
        for (size_t i = 0; i < ENCRYPTED_SIZE; i++)
        {
            const auto c_i =
              get_poly(encrypted, i, coeff_count, coeff_modulus_size);
            set_uint_uint(c_i + next_coeff_modulus_size * coeff_count,
                          coeff_count, temp1);
            auto last_modulus_index = coeff_modulus_size - 1;
            auto last_modulus = coeff_modulus[last_modulus_index];
            uint64_t half = last_modulus >> 1;
            for (size_t j = 0; j < coeff_count; j++)
            {
                temp1[j] =
                  barret_reduce_63(temp1[j] + half, last_modulus,
                                   get_const_ratio(coeff_modulus_const_ratio,
                                                   last_modulus_index));
            }

            for (size_t mod_index = 0; mod_index < next_coeff_modulus_size;
                 mod_index++, temp2_ptr += coeff_count)
            {
                // (ct mod qk) mod qi
                modulo_poly_coeffs_63(
                  temp1, coeff_count, coeff_modulus[mod_index],
                  get_const_ratio(coeff_modulus_const_ratio, mod_index),
                  temp2_ptr);
                // printf("%d\n", half);
                uint64_t half_mod = barret_reduce_63(
                  half, coeff_modulus[mod_index],
                  get_const_ratio(coeff_modulus_const_ratio, mod_index));
                // printf("%d\n", half_mod);
                for (size_t j = 0; j < coeff_count; j++)
                {
                    temp2_ptr[j] = sub_uint_uint_mod(temp2_ptr[j], half_mod,
                                                     coeff_modulus[mod_index]);
                    // printf("%d\n", temp2_ptr[j]);
                }
                // --- seem to work by here

                // ((ct mod qi) - (ct mod qk)) mod qi
                sub_poly_poly_coeffmod(
                  get_poly(encrypted, i, coeff_count, coeff_modulus_size),
                  temp2_ptr, coeff_count, coeff_modulus[mod_index], temp2_ptr);
                // // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                multiply_poly_scalar_coeffmod(
                  temp2_ptr, coeff_count, inv_last_coeff_mod_array[mod_index],
                  coeff_modulus[mod_index],
                  get_const_ratio(coeff_modulus_const_ratio, mod_index),
                  temp2_ptr);
            }
        }

        // set_poly_poly(encrypted, coeff_count * ENCRYPTED_SIZE,
        //               next_coeff_modulus_size, destination)
        set_poly_poly(temp2, coeff_count * ENCRYPTED_SIZE,
                      next_coeff_modulus_size, destination);

        // transform_to_ntt_inplace<<<num_blocks, THREADS_PER_BLOCK>>>(
        //   destination, coeff_modulus, next_coeff_modulus_size, coeff_count,
        //   coeff_count_power, ntt_root_powers, ntt_scaled_root_powers);
        // ::cudaDeviceSynchronize();
        // ::__syncthreads();
    }
}

__device__ void multiply_poly_scalar_coeffmod(
  const uint64_t *poly, size_t coeff_count, uint64_t scalar,
  const uint64_t modulus, const uint64_t_array const_ratio, uint64_t *result)
{
    // Explicit inline
    // for (int i = 0; i < coeff_count; i++)
    //{
    //    *result++ = multiply_uint_uint_mod(*poly++, scalar, modulus);
    //}
    const uint64_t const_ratio_0 = const_ratio[0];
    const uint64_t const_ratio_1 = const_ratio[1];
    for (; coeff_count--; poly++, result++)
    {
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
// NOTE: SOmething weired (occasionaly the value is vary)
__global__ void transform_from_ntt_inplace(
  uint64_t_array encrypted_ntt, // ciphertext
  uint64_t_array coeff_modulus,
  size_t coeff_modulus_count, // coeff modulus
  size_t coeff_count,         // poly_modulus_degree
  int coeff_count_power,      // lg(poly_modulus_degree)
  uint64_t_array ntt_inv_root_powers_div_two,
  uint64_t_array ntt_scaled_inv_root_powers_div_two)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    // const auto poly_uint64_count = coeff_count * coeff_modulus_count;

    if (tid == 0)
    {
        // #pragma unroll
        for (size_t i = 0; i < ENCRYPTED_SIZE; i++)
        {
            for (size_t j = 0; j < coeff_modulus_count; j++)
            {
                inverse_ntt_negacyclic_harvey(
                  get_poly(encrypted_ntt, i, coeff_count, coeff_modulus_count) +
                    (j * coeff_count),
                  coeff_count_power, coeff_modulus[j],
                  ntt_inv_root_powers_div_two + coeff_count * j,
                  ntt_scaled_inv_root_powers_div_two + coeff_count * j);
            }

            // if (tid < coeff_modulus_count)
            // {
            //     // printf("%d\n", tid);
            //     inverse_ntt_negacyclic_harvey(
            //       encrypted_ntt + i * poly_uint64_count + tid *
            //       coeff_count, coeff_count_power, coeff_modulus[tid],
            //       ntt_inv_root_powers_div_two + coeff_count * tid,
            //       ntt_scaled_inv_root_powers_div_two + coeff_count *
            //       tid);
            // }
        }
    }
}

// TODO: change to default cuda kernel(__global__)
__global__ void transform_to_ntt_inplace(
  uint64_t_array encrypted, // ciphertext
  uint64_t_array coeff_modulus,
  size_t coeff_modulus_count, // coeff modulus
  size_t coeff_count,         // poly_modulus_degree
  int coeff_count_power,      // lg(poly_modulus_degree)
  uint64_t_array ntt_root_powers, uint64_t_array ntt_scaled_root_powers)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    // const auto poly_uint64_count = coeff_count * coeff_modulus_count;

    // #pragma unroll
    if (tid == 0)
    {
        for (size_t i = 0; i < ENCRYPTED_SIZE; i++)
        {
            for (size_t j = 0; j < coeff_modulus_count; j++)
            {
                ntt_negacyclic_harvey(
                  get_poly(encrypted, i, coeff_count, coeff_modulus_count) +
                    (j * coeff_count),
                  coeff_count_power, coeff_modulus[j],
                  ntt_root_powers + coeff_count * j,
                  ntt_scaled_root_powers + coeff_count * j);
            }

            // if (tid < coeff_modulus_count)
            // {
            //     ntt_negacyclic_harvey(
            //       encrypted + i * poly_uint64_count + tid * coeff_count,
            //       coeff_count_power, coeff_modulus[tid],
            //       ntt_root_powers + coeff_count * tid,
            //       ntt_scaled_root_powers + coeff_count * tid);
            // }
        }
    }
}

// Inverse negacyclic NTT using Harvey's butterfly. (See Patrick Longa and
// Michael Naehrig).
__device__ void inverse_ntt_negacyclic_harvey_lazy(
  uint64_t_array operand, uint64_t modulus, int coeff_count_power,
  uint64_t_array inv_root_powers_div_two,
  uint64_t_array scaled_inv_root_powers_div_two)
{
    uint64_t two_times_modulus = modulus * 2;

    // return the bit-reversed order of NTT.
    size_t n = size_t(1) << coeff_count_power;
    size_t t = 1;

    // printf("n = %llu, t = %llu\n", n, t);

    for (size_t m = n; m > 1; m >>= 1)
    {
        size_t j1 = 0;
        size_t h = m >> 1;
        // printf("m = %llu, t = %llu, h = %llu\n", m, t, h);
        if (t >= 4)
        {
            for (size_t i = 0; i < h; i++)
            {
                size_t j2 = j1 + t;
                // Need the powers of phi^{-1} in bit-reversed order
                const uint64_t W = inv_root_powers_div_two[h + i];
                const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];
                // printf("\tW = %llu, Wprime = %llu\n", W, Wprime);

                uint64_t *U = operand + j1;
                uint64_t *V = U + t;
                uint64_t currU;
                uint64_t T;
                unsigned long long H;
                for (size_t j = j1; j < j2; j += 4)
                {
                    T = two_times_modulus - *V + *U;
                    currU = *U + *V -
                            (two_times_modulus &
                             static_cast<uint64_t>(
                               -static_cast<int64_t>((*U << 1) >= T)));
                    *U++ =
                      (currU + (modulus & static_cast<uint64_t>(
                                            -static_cast<int64_t>(T & 1)))) >>
                      1;
                    multiply_uint64_hw64(Wprime, T, &H);
                    *V++ = T * W - H * modulus;

                    T = two_times_modulus - *V + *U;
                    currU = *U + *V -
                            (two_times_modulus &
                             static_cast<uint64_t>(
                               -static_cast<int64_t>((*U << 1) >= T)));
                    *U++ =
                      (currU + (modulus & static_cast<uint64_t>(
                                            -static_cast<int64_t>(T & 1)))) >>
                      1;
                    multiply_uint64_hw64(Wprime, T, &H);
                    *V++ = T * W - H * modulus;

                    T = two_times_modulus - *V + *U;
                    currU = *U + *V -
                            (two_times_modulus &
                             static_cast<uint64_t>(
                               -static_cast<int64_t>((*U << 1) >= T)));
                    *U++ =
                      (currU + (modulus & static_cast<uint64_t>(
                                            -static_cast<int64_t>(T & 1)))) >>
                      1;
                    multiply_uint64_hw64(Wprime, T, &H);
                    *V++ = T * W - H * modulus;

                    T = two_times_modulus - *V + *U;
                    currU = *U + *V -
                            (two_times_modulus &
                             static_cast<uint64_t>(
                               -static_cast<int64_t>((*U << 1) >= T)));
                    *U++ =
                      (currU + (modulus & static_cast<uint64_t>(
                                            -static_cast<int64_t>(T & 1)))) >>
                      1;
                    multiply_uint64_hw64(Wprime, T, &H);
                    *V++ = T * W - H * modulus;
                }
                j1 += (t << 1);
            }
        }
        else
        {
            for (size_t i = 0; i < h; i++)
            {
                size_t j2 = j1 + t;
                // Need the powers of  phi^{-1} in bit-reversed order
                const uint64_t W = inv_root_powers_div_two[h + i];
                const uint64_t Wprime = scaled_inv_root_powers_div_two[h + i];

                uint64_t *U = operand + j1;
                uint64_t *V = U + t;
                uint64_t currU;
                uint64_t T;
                unsigned long long H;
                for (size_t j = j1; j < j2; j++)
                {
                    // U = x[i], V = x[i+m]

                    // Compute U - V + 2q
                    T = two_times_modulus - *V + *U;

                    // Cleverly check whether currU + currV >=
                    // two_times_modulus
                    currU = *U + *V -
                            (two_times_modulus &
                             static_cast<uint64_t>(
                               -static_cast<int64_t>((*U << 1) >= T)));

                    // Need to make it so that div2_uint_mod takes values
                    // that are > q. div2_uint_mod(U, modulusptr,
                    // coeff_uint64_count, U); We use also the fact that
                    // parity of currU is same as parity of T. Since our
                    // modulus is always so small that currU +
                    // masked_modulus < 2^64, we never need to worry about
                    // wrapping around when adding masked_modulus. uint64_t
                    // masked_modulus = modulus &
                    // static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
                    // uint64_t carry = add_uint64(currU, masked_modulus, 0,
                    // &currU); currU += modulus &
                    // static_cast<uint64_t>(-static_cast<int64_t>(T & 1));
                    *U++ =
                      (currU + (modulus & static_cast<uint64_t>(
                                            -static_cast<int64_t>(T & 1)))) >>
                      1;

                    multiply_uint64_hw64(Wprime, T, &H);
                    // effectively, the next two multiply perform multiply
                    // modulo beta = 2**wordsize.
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
                                           uint64_t_array scaled_root_powers)
{
    auto two_times_modulus = modulus * 2;

    // Return the NTT in scrambled order
    size_t n = size_t(1) << coeff_count_power;
    size_t t = n >> 1;
    for (size_t m = 1; m < n; m <<= 1)
    {
        if (t >= 4)
        {
            for (size_t i = 0; i < m; i++)
            {
                size_t j1 = 2 * i * t;
                size_t j2 = j1 + t;
                const uint64_t W = root_powers[m + i];
                const uint64_t Wprime = scaled_root_powers[m + i];

                uint64_t *X = operand + j1;
                uint64_t *Y = X + t;
                uint64_t currX;
                unsigned long long Q;
                for (size_t j = j1; j < j2; j += 4)
                {
                    currX = *X - (two_times_modulus &
                                  static_cast<uint64_t>(-static_cast<int64_t>(
                                    *X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus &
                                  static_cast<uint64_t>(-static_cast<int64_t>(
                                    *X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus &
                                  static_cast<uint64_t>(-static_cast<int64_t>(
                                    *X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);

                    currX = *X - (two_times_modulus &
                                  static_cast<uint64_t>(-static_cast<int64_t>(
                                    *X >= two_times_modulus)));
                    multiply_uint64_hw64(Wprime, *Y, &Q);
                    Q = *Y * W - Q * modulus;
                    *X++ = currX + Q;
                    *Y++ = currX + (two_times_modulus - Q);
                }
            }
        }
        else
        {
            for (size_t i = 0; i < m; i++)
            {
                size_t j1 = 2 * i * t;
                size_t j2 = j1 + t;
                const uint64_t W = root_powers[m + i];
                const uint64_t Wprime = scaled_root_powers[m + i];

                uint64_t *X = operand + j1;
                uint64_t *Y = X + t;
                uint64_t currX;
                unsigned long long Q;
                for (size_t j = j1; j < j2; j++)
                {
                    // The Harvey butterfly: assume X, Y in [0, 2p), and
                    // return X', Y' in [0, 2p). X', Y' = X + WY, X - WY
                    // (mod p).
                    currX = *X - (two_times_modulus &
                                  static_cast<uint64_t>(-static_cast<int64_t>(
                                    *X >= two_times_modulus)));
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

__global__ void kernel(void)
{
    printf("Hello from CUDA kernel.\n");
}

void proxy(void)
{
    kernel<<<1, 1>>>();
}
