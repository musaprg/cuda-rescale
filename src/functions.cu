//
// Created by kotaro on 2019/12/13.
//

#include "functions.h"
#include <cstdio>

/*
 * 1. pour data from Ciphertext
 * 2. memcpy to GPU
 * 3. perform evaluation
 * 4. send back to host
 * 5. Update Ciphertext instances
 */

// ---------------------------------------------------------------------------

void rescale_to_next(const CuCiphertext &encrypted, CuCiphertext &destination,
                     const CudaContextData &context) {
  // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA%A4%C7%B9%D4%CE%F3%B1%E9%BB%BB%A1%A7%B2%C3%B8%BA%BB%BB

  size_t encrypted_size = encrypted.size();
  // TODO: precaliculate destination size from input parameters.
  size_t destination_size = destination.size();
  auto device_encrypted = cuda::make_unique<uint64_t[]>(encrypted_size);
  auto device_destination = cuda::make_unique<uint64_t[]>(destination_size);
  auto device_context = cuda::make_unique<CudaContextData>();

  ::cudaMemcpy(device_encrypted.get(), &encrypted[0],
               sizeof(uint64_t) * encrypted_size, cudaMemcpyDefault);
  ::cudaMemcpy(device_context.get(), &context, sizeof(CudaContextData),
               cudaMemcpyDefault);
  ::cudaDeviceSynchronize(); // synchronize

  mod_switch_scale_to_next<<<1, 1>>>(
      device_encrypted.get(), device_destination.get(), device_context.get());
  //  ::cudaDeviceSynchronize();

  ::cudaMemcpy(&destination[0], device_destination.get(), destination_size,
               cudaMemcpyDefault);
}

__global__ void mod_switch_scale_to_next(uint64_t_array encrypted,
                                         uint64_t_array destination,
                                         CudaContextData *context) {}

__global__ void kernel(void) { printf("Hello from CUDA kernel.\n"); }

void proxy(void) { kernel<<<1, 1>>>(); }
