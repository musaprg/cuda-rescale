//
// Created by kotaro on 2019/12/22.
//

#pragma once

#include <vector>

using namespace std;

// datas of Context will be integrated into this struct.
// some data which cuda kernels use are picked up and stored.
struct CudaContextData {
  vector<uint64_t> coeff_modulus;
  vector<uint64_t> next_coeff_modulus;
  vector<uint64_t> current_ntt_tables;
  vector<uint64_t> next_ntt_tables;

  CudaContextData(vector<uint64_t> coeff_modulus,
                  vector<uint64_t> next_coeff_modulus)
      : coeff_modulus(move(coeff_modulus)),
        next_coeff_modulus(move(next_coeff_modulus)){};
};