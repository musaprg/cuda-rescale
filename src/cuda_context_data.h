//
// Created by kotaro on 2019/12/22.
//

#pragma once

#include <vector>

using namespace std;

// datas of Context will be integrated into this struct.
// some data which cuda kernels use are picked up and stored.
struct CudaContextData {
  //  vector<uint64_t> coeff_modulus;
  uint64_t *coeff_modulus;
  size_t coeff_modulus_size;
  //  vector<uint64_t> next_coeff_modulus;
  uint64_t *next_coeff_modulus;
  size_t next_coeff_modulus_size;
  //  vector<uint64_t> current_ntt_tables;
  //  vector<uint64_t> next_ntt_tables;

  //  CudaContextData(vector<uint64_t> coeff_modulus,
  //                  vector<uint64_t> next_coeff_modulus)
  //      : coeff_modulus(move(coeff_modulus)),
  //        next_coeff_modulus(move(next_coeff_modulus)){};

  CudaContextData(vector<uint64_t> coeff_modulus,
                  vector<uint64_t> next_coeff_modulus)
      : coeff_modulus(coeff_modulus.data()),
        coeff_modulus_size(coeff_modulus.size()),
        next_coeff_modulus(next_coeff_modulus.data()),
        next_coeff_modulus_size(next_coeff_modulus.size()) {}
};