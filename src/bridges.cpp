//
// Created by kotaro on 2019/12/22.
//

#include "bridges.h"

/*
 * bridges class aims to bridge data transformations between host and
 * CUDA-enabled headers. This class is created to avoid compilation errors due
 * to c++ version conflict.
 */

CudaContextData
get_cuda_context_data(const shared_ptr<seal::SEALContext> &context,
                      const seal::Ciphertext &encrypted,
                      const seal::Ciphertext &destination) {
  // TODO: parms_id may be gotten from context? e.g. context.first_parm_id()
  auto context_data_ptr = context->get_context_data(encrypted.parms_id());

  // picked from Evaluator::mod_switch_scale_to_next(const Ciphertext
  // &encrypted, Ciphertext &destination, MemoryPoolHandle pool) Extract
  // encryption parameters.
  auto &context_data = *context_data_ptr;
  auto &next_context_data = *context_data.next_context_data();
  auto &next_parms = next_context_data.parms();

  // q_1,...,q_{k-1}
  auto coeff_modulus = convert_small_modulus_vec_to_uint_vec(
      context_data.parms().coeff_modulus());
  auto next_coeff_modulus =
      convert_small_modulus_vec_to_uint_vec(next_parms.coeff_modulus());
  size_t next_coeff_mod_count = next_coeff_modulus.size();
  size_t coeff_count = next_parms.poly_modulus_degree();
  size_t encrypted_size = encrypted.size();
  auto &inv_last_coeff_mod_array =
      context_data.base_converter()->get_inv_last_coeff_mod_array();

  auto last_modulus = context_data.parms().coeff_modulus().back();

  //  auto current_ntt_table = context_data.small_ntt_tables()->modulus();
  //  auto next_ntt_table = next_context_data.small_ntt_tables()->modulus();

  CudaContextData cuda_context_data(coeff_modulus, next_coeff_modulus);

  return cuda_context_data;
}