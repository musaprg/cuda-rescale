//
// Created by kotaro on 2019/12/22.
//

#pragma once

#include <memory>

#include "cuda_context_data.h"
#include "seal/seal.h"

using namespace std;

CudaContextData get_cuda_context_data(shared_ptr<seal::SEALContext> context);

vector<uint64_t>
convert_small_modulus_vec_to_uint_vec(vector<seal::SmallModulus> src) {
  vector<uint64_t> ret;
  ret.reserve(src.size());

  for (auto &&v : src) {
    ret.push_back(v.value());
  }

  return ret;
}