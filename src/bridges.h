//
// Created by kotaro on 2019/12/22.
//

#pragma once

#include <memory>

#include "cuda_context_data.h"
// #include "functions.h"
#include "seal/seal.h"

using namespace std;

// TODO: duplicated to functions.h, refine these code.
using CuCiphertext = vector<uint64_t>;

inline CuCiphertext get_cuciphertext_from_ciphertext(
  const seal::Ciphertext &ciphertext)
{
    CuCiphertext ret;
    auto source_size = ciphertext.uint64_count();
    ret.reserve(source_size);

    auto it = ciphertext.data();
    for (size_t i = 0; i < source_size; ++i)
    {
        ret.emplace_back(*it++);
    }

    return ret;
}

CudaContextData get_cuda_context_data(const shared_ptr<seal::SEALContext> &,
                                      const seal::Ciphertext &,
                                      const seal::Ciphertext &);

// TODO: Use this when refactoring (for decreasing the number of transmission)
/*
 * ret[current_level *]
 * [invq_0 mod q_1, invq_1 mod q_1, invq_0 mod q_2, invq_1 mod q_2, ...]
 */
inline vector<uint64_t> get_inv_last_coeff_mod_array_from_context(
  const shared_ptr<seal::SEALContext> &context)
{
    cout << "[[get inv_last_coeff_mod_array]]" << endl;
    auto context_data_ptr = context->first_context_data();

    vector<uint64_t> ret;

    while (context_data_ptr->next_context_data())
    {
        auto &context_data = *context_data_ptr;
        cout << "Level (chain_index) " << context_data.chain_index() << endl;
        auto coeff_base_mod_count =
          context_data.base_converter()->coeff_base_mod_count();
        auto &inv_last_coeff_mod_array =
          context_data.base_converter()->get_inv_last_coeff_mod_array();

        for (size_t i = 0; i < coeff_base_mod_count; i++)
        {
            cout << inv_last_coeff_mod_array[i] << " ";
            ret.push_back(inv_last_coeff_mod_array[i]);
        }
        cout << endl;

        context_data_ptr = context_data_ptr->next_context_data();
    }

    return ret;
}

inline vector<uint64_t> get_inv_last_coeff_mod_array_from_encrypted(
  const shared_ptr<seal::SEALContext> &context,
  const seal::Ciphertext &encrypted)
{
    cout << "[[get inv_last_coeff_mod_array]]" << endl;

    auto context_data_ptr = context->get_context_data(encrypted.parms_id());
    auto &context_data = *context_data_ptr;
    auto coeff_base_mod_count =
      context_data.base_converter()->coeff_base_mod_count();
    auto &inv_last_coeff_mod_array =
      context_data.base_converter()->get_inv_last_coeff_mod_array();
    cout << "Ciphertext Level (chain_index) " << context_data.chain_index()
         << endl;

    vector<uint64_t> ret;

    for (size_t i = 0; i < coeff_base_mod_count; i++)
    {
        cout << inv_last_coeff_mod_array[i] << " ";
        ret.push_back(inv_last_coeff_mod_array[i]);
    }
    cout << endl;

    return ret;
}

inline vector<uint64_t> convert_small_modulus_vec_to_uint_vec(
  const vector<seal::SmallModulus> &src)
{
    vector<uint64_t> ret;
    // ret.reserve(src.size());

    for (auto &&v : src)
    {
        ret.push_back(v.value());
    }

    return ret;
}

// {q_1_ratio, q_2_ratio, q_3_ratio, ... }
inline vector<uint64_t> convert_small_modulus_coeff_ratio_to_uint_vec(
  const vector<seal::SmallModulus> &src)
{
    vector<uint64_t> ret;
    // ret.reserve(src.size() * 3);

    for (auto &&v : src)
    {
        for (auto &&cr : v.const_ratio())
        {
            ret.push_back(cr);
        }
    }

    return ret;
}

// convert seal::util::SmallNTTTables into primitive std::vector
inline void convert_small_ntt_tables_vec_to_uint_vec(
  const shared_ptr<seal::SEALContext> &context,
  //    const seal::util::Pointer<seal::util::SmallNTTTables>
  //    &small_ntt_tables,
  // vector<vector<uint64_t>> &ntt_root_powers, // 2d array version
  // vector<vector<uint64_t>> &ntt_scaled_root_powers,
  // vector<vector<uint64_t>> &ntt_inv_root_powers_div_two,
  // vector<vector<uint64_t>> &ntt_scaled_inv_root_powers_div_two
  vector<uint64_t> &ntt_root_powers, vector<uint64_t> &ntt_scaled_root_powers,
  vector<uint64_t> &ntt_inv_root_powers_div_two,
  vector<uint64_t> &ntt_scaled_inv_root_powers_div_two)
{
    auto &context_data = *context->first_context_data();
    auto &parms = context_data.parms();
    size_t coeff_count = parms.poly_modulus_degree();
    size_t coeff_mod_count = parms.coeff_modulus().size();
    auto &coeff_small_ntt_tables = context_data.small_ntt_tables();

    cout << "coeff_count: " << coeff_count << endl;
    cout << "coeff_mod_count: " << coeff_mod_count << endl;

    // 2d array version
    // ntt_root_powers.resize(coeff_mod_count, vector<uint64_t>(coeff_count));
    // ntt_scaled_root_powers.resize(coeff_mod_count,
    //                               vector<uint64_t>(coeff_count));
    // ntt_inv_root_powers_div_two.resize(coeff_mod_count,
    //                                    vector<uint64_t>(coeff_count));
    // ntt_scaled_inv_root_powers_div_two.resize(coeff_mod_count,
    //                                           vector<uint64_t>(coeff_count));

    ntt_root_powers.resize(coeff_mod_count * coeff_count);
    ntt_scaled_root_powers.resize(coeff_mod_count * coeff_count);
    ntt_inv_root_powers_div_two.resize(coeff_mod_count * coeff_count);
    ntt_scaled_inv_root_powers_div_two.resize(coeff_mod_count * coeff_count);

    // TODO: If there's something weird, check these line.
    for (size_t i = 0; i < coeff_mod_count; i++)
    {
        auto &small_ntt_table = coeff_small_ntt_tables[i];
        cout << "[q_" << i << "] "
             << "coeff_count: " << small_ntt_table.coeff_count() << endl;
        cout << "[q_" << i << "] "
             << "coeff_count_power: " << small_ntt_table.coeff_count_power()
             << endl;
        for (size_t j = 0; j < coeff_count; j++)
        {
            auto root_power = small_ntt_table.get_from_root_powers(j);
            auto inv_root_power_div_two =
              small_ntt_table.get_from_inv_root_powers_div_two(j);
            auto scaled_root_power =
              small_ntt_table.get_from_scaled_root_powers(j);
            auto scaled_inv_root_power_div_two =
              small_ntt_table.get_from_scaled_inv_root_powers_div_two(j);

            // cout << root_power << " " << inv_root_power_div_two << " "
            //      << scaled_root_power << " " << scaled_inv_root_power_div_two
            //      << " " << endl;

            // 2d array version
            // ntt_root_powers[i].emplace_back(root_power);
            // ntt_inv_root_powers_div_two[i].emplace_back(inv_root_power_div_two);
            // ntt_scaled_root_powers[i].emplace_back(scaled_root_power);
            // ntt_scaled_inv_root_powers_div_two[i].emplace_back(
            //   scaled_inv_root_power_div_two);

            ntt_root_powers.emplace_back(root_power);
            ntt_inv_root_powers_div_two.emplace_back(inv_root_power_div_two);
            ntt_scaled_root_powers.emplace_back(scaled_root_power);
            ntt_scaled_inv_root_powers_div_two.emplace_back(
              scaled_inv_root_power_div_two);
        }
    }
}