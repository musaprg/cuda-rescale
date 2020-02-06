//
// Created by kotaro on 2019/12/22.
//

#pragma once

#include <vector>

using namespace std;

// datas of Context will be integrated into this struct.
// some data which cuda kernels use are picked up and stored.
struct CudaContextData
{
    //  vector<uint64_t> coeff_modulus;
    vector<uint64_t> coeff_modulus;
    vector<uint64_t> coeff_modulus_const_ratio;
    //  size_t coeff_modulus_size;
    //  vector<uint64_t> next_coeff_modulus;
    vector<uint64_t> next_coeff_modulus;
    vector<uint64_t> next_coeff_modulus_const_ratio;
    vector<uint64_t> inv_last_coeff_mod_array;
    size_t coeff_count; // poly_modulus_degree
    int coeff_count_power;

    // SmallNTTTables
    // vector<vector<uint64_t>> ntt_root_powers;
    // vector<vector<uint64_t>> ntt_scaled_root_powers;
    // vector<vector<uint64_t>> ntt_inv_root_powers_div_two;
    // vector<vector<uint64_t>> ntt_scaled_inv_root_powers_div_two;
    vector<uint64_t> ntt_root_powers;
    vector<uint64_t> ntt_scaled_root_powers;
    vector<uint64_t> ntt_inv_root_powers_div_two;
    vector<uint64_t> ntt_scaled_inv_root_powers_div_two;

    CudaContextData(vector<uint64_t> coeff_modulus,
                    vector<uint64_t> coeff_modulus_const_ratio,
                    vector<uint64_t> next_coeff_modulus,
                    vector<uint64_t> next_coeff_modulus_const_ratio,
                    vector<uint64_t> inv_last_coeff_mod_array,
                    size_t coeff_count, int coeff_count_power,
                    vector<uint64_t> ntt_root_powers,
                    vector<uint64_t> ntt_scaled_root_powers,
                    vector<uint64_t> ntt_inv_root_powers_div_two,
                    vector<uint64_t> ntt_scaled_inv_root_powers_div_two)
      : coeff_modulus(coeff_modulus),
        coeff_modulus_const_ratio(coeff_modulus_const_ratio),
        next_coeff_modulus(next_coeff_modulus),
        next_coeff_modulus_const_ratio(next_coeff_modulus_const_ratio),
        inv_last_coeff_mod_array(inv_last_coeff_mod_array),
        coeff_count(coeff_count),
        coeff_count_power(coeff_count_power),
        ntt_root_powers(ntt_root_powers),
        ntt_scaled_root_powers(ntt_scaled_root_powers),
        ntt_inv_root_powers_div_two(ntt_inv_root_powers_div_two),
        ntt_scaled_inv_root_powers_div_two(ntt_scaled_inv_root_powers_div_two)
    {
    }

    // CudaContextData(vector<uint64_t> coeff_modulus,
    //                 vector<uint64_t> next_coeff_modulus, size_t coeff_count,
    //                 vector<vector<uint64_t>> ntt_root_powers,
    //                 vector<vector<uint64_t>> ntt_scaled_root_powers,
    //                 vector<vector<uint64_t>> ntt_inv_root_powers_div_two,
    //                 vector<vector<uint64_t>>
    //                 ntt_scaled_inv_root_powers_div_two)
    //   : coeff_modulus(coeff_modulus),
    //     next_coeff_modulus(next_coeff_modulus),
    //     coeff_count(coeff_count),
    //     ntt_root_powers(ntt_root_powers),
    //     ntt_scaled_root_powers(ntt_scaled_root_powers),
    //     ntt_inv_root_powers_div_two(ntt_inv_root_powers_div_two),
    //     ntt_scaled_inv_root_powers_div_two(ntt_scaled_inv_root_powers_div_two)
    // {
    // }
};