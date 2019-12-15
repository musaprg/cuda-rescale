//
// Created by kotaro on 2019/12/13.
//

#include <cstdio>
#include "functions.h"

/*
 * 1. pour data from Ciphertext
 * 2. memcpy to GPU
 * 3. perform evaluation
 * 4. send back to host
 * 5. Update Ciphertext instances
 */

__global__ void kernel(void) {
  printf("Hello from CUDA kernel.\n");
}

void proxy(void) {
  kernel<<<1,1>>>();
}