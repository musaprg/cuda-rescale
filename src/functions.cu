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

namespace cuda {} // namespace cuda

void initialize(CudaContextData &contextdata) {}

void cleanup() {
  // cuda::data = CudaContextData{};
}

// ---------------------------------------------------------------------------

void rescale_to_next(const CuCiphertext &encrypted, CuCiphertext &destination) {
  // http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?CUDA%A4%C7%B9%D4%CE%F3%B1%E9%BB%BB%A1%A7%B2%C3%B8%BA%BB%BB

  size_t encrypted_size = encrypted.size();
  // TODO: precaliculate destination size from input parameters.
  size_t destination_size = destination.size();
  auto device_encrypted = cuda::make_unique<uint64_t[]>(encrypted_size);
  auto device_destination = cuda::make_unique<uint64_t[]>(destination_size);

  ::cudaMemcpy(device_encrypted.get(), &encrypted[0], encrypted_size,
               cudaMemcpyDefault);
  ::cudaMemcpy(device_destination.get(), &destination[0], destination_size,
               cudaMemcpyDefault);
  ::cudaDeviceSynchronize(); // synchronize

  mod_switch_scale_to_next(device_encrypted.get(), device_destination.get());
  ::cudaDeviceSynchronize();

  ::cudaMemcpy(&destination[0], (void **)&device_destination, destination_size)
}

__global__ void mod_switch_scale_to_next(uint64_t_array encrypted,
                                         uint64_t_array destination) {}

__global__ void kernel(void) { printf("Hello from CUDA kernel.\n"); }

void proxy(void) { kernel<<<1, 1>>>(); }
