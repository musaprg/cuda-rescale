//
// Created by kotaro on 2019/12/22.
//

#pragma once

#include <memory>

#include "cuda_context_data.h"
#include "seal/seal.h"

using namespace std;

using CuCiphertext = vector<uint64_t>;

inline CuCiphertext get_cuciphertext_from_ciphertext(const seal::Ciphertext &ciphertext)
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

inline vector<uint64_t> convert_small_modulus_vec_to_uint_vec(
  const vector<seal::SmallModulus> &src)
{
    vector<uint64_t> ret;
    ret.reserve(src.size());

    for (auto &&v : src)
    {
        ret.push_back(v.value());
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