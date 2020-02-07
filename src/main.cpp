// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Modified by Kotaro Inoue <kinoue@yama.info.waseda.ac.jp>

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#include <cxxabi.h>
#endif

#include "bridges.h"
#include "examples.hpp"
#include "functions.h"
#include "seal/seal.h"
#include "timer.hpp"

using namespace std;

void validate_implementation()
{
    seal::EncryptionParameters parms(seal::scheme_type::CKKS);

    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
      seal::CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60})); // L2

    double scale = pow(2.0, 40);

    auto context = seal::SEALContext::Create(parms);
    print_parameters(context);
    cout << endl;

    seal::KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    seal::Encryptor encryptor(context, public_key);

    seal::CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    seal::Evaluator evaluator(context);

    vector<double> input;
    input.reserve(slot_count);
    double curr_point = 0;
    double step_size = 1.0 / (static_cast<double>(slot_count) - 1);
    for (size_t i = 0; i < slot_count; i++, curr_point += step_size)
    {
        input.push_back(curr_point);
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

    cout << "Evaluating polynomial PI*x^3 + 0.4x + 1 ..." << endl;

    seal::Plaintext x_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(input, scale, x_plain);
    seal::Ciphertext x1_encrypted;
    encryptor.encrypt(x_plain, x1_encrypted);

    auto context_data_ptr = context->get_context_data(x1_encrypted.parms_id());
    auto &context_data = *context_data_ptr;
    auto &next_context_data = *context_data.next_context_data();
    auto &next_parms = next_context_data.parms();

#ifndef NDEBUG
    {
        cout << "[[Check build-in iNTT]]" << endl;
        seal::Ciphertext destination;
        auto encrypted_cu = get_cuciphertext_from_ciphertext(x1_encrypted);
        for (size_t i = 0; i < 5; i++)
        {
            cout << encrypted_cu.at(i) << " ";
        }
        cout << endl;
        evaluator.transform_from_ntt(x1_encrypted, destination);
        encrypted_cu = get_cuciphertext_from_ciphertext(destination);
        for (size_t i = 0; i < 5; i++)
        {
            cout << encrypted_cu.at(i) << " ";
        }
        cout << endl;
    }
#endif

    auto x1_encrypted_cu = get_cuciphertext_from_ciphertext(x1_encrypted);

    auto destination = x1_encrypted;
    auto destination_cu = get_cuciphertext_from_ciphertext(destination);

    CudaContextData cucontext =
      get_cuda_context_data(context, x1_encrypted, destination);

    cout << "Used device: " << get_device_name(0) << endl;
#ifndef NDEBUG
    cout << "Before rescale vector size: " << destination_cu.size() << endl;
#endif
    rescale_to_next(x1_encrypted_cu, destination_cu, cucontext);

    {
        seal::Ciphertext after_rescale;
        evaluator.rescale_to_next(x1_encrypted, after_rescale);

#ifndef NDEBUG
        cout << "After rescale vector size:" << destination_cu.size() << endl;
        cout << "After rescale vector size(correct): "
             << after_rescale.uint64_count() << endl;

        size_t wrong_coeff_count = 0;
        size_t no_affect_coeff_count = 0;
        for (size_t i = 0; i < destination_cu.size(); i++)
        {
            if (destination_cu.at(i) != after_rescale[i])
            {
                if (destination_cu.at(i) == destination[i])
                {
                    no_affect_coeff_count++;
                    // cout << "[No Affect at " << i << "] ";
                }
                else
                {
                    wrong_coeff_count++;
                    // cout << "[Wrong at " << i << "] ";
                }
                // cout << destination_cu.at(i)
                //      << "| expected: " << after_rescale[i]
                //      << " | before: " << destination[i] << endl;
            }
        }
        cout << "Total wrong coeff count: " << wrong_coeff_count << "/"
             << destination_cu.size() << endl;
        cout << "Total no affect coeff count: " << no_affect_coeff_count << "/"
             << destination_cu.size() << endl;
#else
        for (size_t i = 0; i < destination_cu.size(); i++)
        {
            assert(destination_cu.at(i) == after_rescale[i]);
        }
#endif
    }
}

int main()
{
    validate_implementation();

    return 0;
}
