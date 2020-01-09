// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
//
// Modified by Kotaro Inoue <kinoue@yama.info.waseda.ac.jp>

#include <memory>
#include <string>
#include <cstdlib>
#include <type_traits>
#include <typeinfo>
#ifndef _MSC_VER
#   include <cxxabi.h>
#endif

#include "examples.hpp"
#include "functions.h"
#include "seal/cudaevaluator.h"
#include "timer.hpp"

using namespace std;
using namespace seal;

/*template <class T>
std::string
type_name()
{
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void(*)(void*)> own
      (
#ifndef _MSC_VER
      abi::__cxa_demangle(typeid(TR).name(), nullptr,
                          nullptr, nullptr),
#else
      nullptr,
#endif
      std::free
  );
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value)
    r += " const";
  if (std::is_volatile<TR>::value)
    r += " volatile";
  if (std::is_lvalue_reference<T>::value)
    r += "&";
  else if (std::is_rvalue_reference<T>::value)
    r += "&&";
  return r;
}*/

// https://github.com/microsoft/SEAL/blob/master/native/examples/4_ckks_basics.cpp
// Fixed by @musaprg
void example_ckks_basics() {
  print_example_banner("Example: CKKS Basics");

  EncryptionParameters parms(scheme_type::CKKS);

  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  double scale = pow(2.0, 40);

    auto context = SEALContext::Create(parms);
    print_parameters(context);
    cout << endl;

    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

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

    /*
    We create plaintexts for PI, 0.4, and 1 using an overload of CKKSEncoder::encode
    that encodes the given floating-point value to every slot in the vector.
    */
    Plaintext plain_coeff3, plain_coeff1, plain_coeff0;
    encoder.encode(3.14159265, scale, plain_coeff3);
    encoder.encode(0.4, scale, plain_coeff1);
    encoder.encode(1.0, scale, plain_coeff0);

    Plaintext x_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(input, scale, x_plain);
    Ciphertext x1_encrypted;
    encryptor.encrypt(x_plain, x1_encrypted);

    /*
    To compute x^3 we first compute x^2 and relinearize. However, the scale has
    now grown to 2^80.
    */
    Ciphertext x3_encrypted;
    print_line(__LINE__);
    cout << "Compute x^2 and relinearize:" << endl;
    evaluator.square(x1_encrypted, x3_encrypted);
    evaluator.relinearize_inplace(x3_encrypted, relin_keys);
    cout << "    + Scale of x^2 before rescale: " << log2(x3_encrypted.scale())
        << " bits" << endl;

    /*
    Now rescale; in addition to a modulus switch, the scale is reduced down by
    a factor equal to the prime that was switched away (40-bit prime). Hence, the
    new scale should be close to 2^40. Note, however, that the scale is not equal
    to 2^40: this is because the 40-bit prime is only close to 2^40.
    */
    print_line(__LINE__);
    cout << "Rescale x^2." << endl;
    evaluator.rescale_to_next_inplace(x3_encrypted);
    cout << "    + Scale of x^2 after rescale: " << log2(x3_encrypted.scale())
        << " bits" << endl;

    /*
    Now x3_encrypted is at a different level than x1_encrypted, which prevents us
    from multiplying them to compute x^3. We could simply switch x1_encrypted to
    the next parameters in the modulus switching chain. However, since we still
    need to multiply the x^3 term with PI (plain_coeff3), we instead compute PI*x
    first and multiply that with x^2 to obtain PI*x^3. To this end, we compute
    PI*x and rescale it back from scale 2^80 to something close to 2^40.
    */
    print_line(__LINE__);
    cout << "Compute and rescale PI*x." << endl;
    Ciphertext x1_encrypted_coeff3;
    evaluator.multiply_plain(x1_encrypted, plain_coeff3, x1_encrypted_coeff3);
    cout << "    + Scale of PI*x before rescale: " << log2(x1_encrypted_coeff3.scale())
        << " bits" << endl;
    evaluator.rescale_to_next_inplace(x1_encrypted_coeff3);
    cout << "    + Scale of PI*x after rescale: " << log2(x1_encrypted_coeff3.scale())
        << " bits" << endl;

    /*
    Since x3_encrypted and x1_encrypted_coeff3 have the same exact scale and use
    the same encryption parameters, we can multiply them together. We write the
    result to x3_encrypted, relinearize, and rescale. Note that again the scale
    is something close to 2^40, but not exactly 2^40 due to yet another scaling
    by a prime. We are down to the last level in the modulus switching chain.
    */
    print_line(__LINE__);
    cout << "Compute, relinearize, and rescale (PI*x)*x^2." << endl;
    evaluator.multiply_inplace(x3_encrypted, x1_encrypted_coeff3);
    evaluator.relinearize_inplace(x3_encrypted, relin_keys);
    cout << "    + Scale of PI*x^3 before rescale: " << log2(x3_encrypted.scale())
        << " bits" << endl;
    evaluator.rescale_to_next_inplace(x3_encrypted);
    cout << "    + Scale of PI*x^3 after rescale: " << log2(x3_encrypted.scale())
        << " bits" << endl;

    /*
    Next we compute the degree one term. All this requires is one multiply_plain
    with plain_coeff1. We overwrite x1_encrypted with the result.
    */
    print_line(__LINE__);
    cout << "Compute and rescale 0.4*x." << endl;
    evaluator.multiply_plain_inplace(x1_encrypted, plain_coeff1);
    cout << "    + Scale of 0.4*x before rescale: " << log2(x1_encrypted.scale())
        << " bits" << endl;
    evaluator.rescale_to_next_inplace(x1_encrypted);
    cout << "    + Scale of 0.4*x after rescale: " << log2(x1_encrypted.scale())
        << " bits" << endl;

    /*
    Now we would hope to compute the sum of all three terms. However, there is
    a serious problem: the encryption parameters used by all three terms are
    different due to modulus switching from rescaling.

    Encrypted addition and subtraction require that the scales of the inputs are
    the same, and also that the encryption parameters (parms_id) match. If there
    is a mismatch, Evaluator will throw an exception.
    */
    cout << endl;
    print_line(__LINE__);
    cout << "Parameters used by all three terms are different." << endl;
    cout << "    + Modulus chain index for x3_encrypted: "
        << context->get_context_data(x3_encrypted.parms_id())->chain_index() << endl;
    cout << "    + Modulus chain index for x1_encrypted: "
        << context->get_context_data(x1_encrypted.parms_id())->chain_index() << endl;
    cout << "    + Modulus chain index for plain_coeff0: "
        << context->get_context_data(plain_coeff0.parms_id())->chain_index() << endl;
    cout << endl;

    /*
    Let us carefully consider what the scales are at this point. We denote the
    primes in coeff_modulus as P_0, P_1, P_2, P_3, in this order. P_3 is used as
    the special modulus and is not involved in rescalings. After the computations
    above the scales in ciphertexts are:

        - Product x^2 has scale 2^80 and is at level 2;
        - Product PI*x has scale 2^80 and is at level 2;
        - We rescaled both down to scale 2^80/P_2 and level 1;
        - Product PI*x^3 has scale (2^80/P_2)^2;
        - We rescaled it down to scale (2^80/P_2)^2/P_1 and level 0;
        - Product 0.4*x has scale 2^80;
        - We rescaled it down to scale 2^80/P_2 and level 1;
        - The contant term 1 has scale 2^40 and is at level 2.

    Although the scales of all three terms are approximately 2^40, their exact
    values are different, hence they cannot be added together.
    */
    print_line(__LINE__);
    cout << "The exact scales of all three terms are different:" << endl;
    ios old_fmt(nullptr);
    old_fmt.copyfmt(cout);
    cout << fixed << setprecision(10);
    cout << "    + Exact scale in PI*x^3: " << x3_encrypted.scale() << endl;
    cout << "    + Exact scale in  0.4*x: " << x1_encrypted.scale() << endl;
    cout << "    + Exact scale in      1: " << plain_coeff0.scale() << endl;
    cout << endl;
    cout.copyfmt(old_fmt);

    /*
    There are many ways to fix this problem. Since P_2 and P_1 are really close
    to 2^40, we can simply "lie" to Microsoft SEAL and set the scales to be the
    same. For example, changing the scale of PI*x^3 to 2^40 simply means that we
    scale the value of PI*x^3 by 2^120/(P_2^2*P_1), which is very close to 1.
    This should not result in any noticeable error.

    Another option would be to encode 1 with scale 2^80/P_2, do a multiply_plain
    with 0.4*x, and finally rescale. In this case we would need to additionally
    make sure to encode 1 with appropriate encryption parameters (parms_id).

    In this example we will use the first (simplest) approach and simply change
    the scale of PI*x^3 and 0.4*x to 2^40.
    */
    print_line(__LINE__);
    cout << "Normalize scales to 2^40." << endl;
    x3_encrypted.scale() = pow(2.0, 40);
    x1_encrypted.scale() = pow(2.0, 40);

    /*
    We still have a problem with mismatching encryption parameters. This is easy
    to fix by using traditional modulus switching (no rescaling). CKKS supports
    modulus switching just like the BFV scheme, allowing us to switch away parts
    of the coefficient modulus when it is simply not needed.
    */
    print_line(__LINE__);
    cout << "Normalize encryption parameters to the lowest level." << endl;
    parms_id_type last_parms_id = x3_encrypted.parms_id();
    evaluator.mod_switch_to_inplace(x1_encrypted, last_parms_id);
    evaluator.mod_switch_to_inplace(plain_coeff0, last_parms_id);

    /*
    All three ciphertexts are now compatible and can be added.
    */
    print_line(__LINE__);
    cout << "Compute PI*x^3 + 0.4*x + 1." << endl;
    Ciphertext encrypted_result;
    evaluator.add(x3_encrypted, x1_encrypted, encrypted_result);
    evaluator.add_plain_inplace(encrypted_result, plain_coeff0);

    /*
    First print the true result.
    */
    Plaintext plain_result;
    print_line(__LINE__);
    cout << "Decrypt and decode PI*x^3 + 0.4x + 1." << endl;
    cout << "    + Expected result:" << endl;
    vector<double> true_result;
    for (size_t i = 0; i < input.size(); i++)
    {
        double x = input[i];
        true_result.push_back((3.14159265 * x * x + 0.4)* x + 1);
    }
    print_vector(true_result, 3, 7);

    /*
    Decrypt, decode, and print the result.
    */
    decryptor.decrypt(encrypted_result, plain_result);
    vector<double> result;
    encoder.decode(plain_result, result);
    cout << "    + Computed result ...... Correct." << endl;
    print_vector(result, 3, 7);

    /*
    While we did not show any computations on complex numbers in these examples,
    the CKKSEncoder would allow us to have done that just as easily. Additions
    and multiplications of complex numbers behave just as one would expect.
    */
}

// https://github.com/microsoft/SEAL/blob/master/native/examples/6_performance.cpp
// Fixed by @musaprg
void ckks_performance_test(shared_ptr<SEALContext> context) {
  chrono::high_resolution_clock::time_point time_start, time_end;

  print_parameters(context);
  cout << endl;

  auto &parms = context->first_context_data()->parms();
  size_t poly_modulus_degree = parms.poly_modulus_degree();

  cout << "Generating secret/public keys: ";
  KeyGenerator keygen(context);
  cout << "Done" << endl;

  auto secret_key = keygen.secret_key();
  auto public_key = keygen.public_key();

  RelinKeys relin_keys;
  GaloisKeys gal_keys;
  chrono::microseconds time_diff;
  if (context->using_keyswitching()) {
    cout << "Generating relinearization keys: ";
    time_start = chrono::high_resolution_clock::now();
    relin_keys = keygen.relin_keys();
    time_end = chrono::high_resolution_clock::now();
    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]" << endl;

    if (!context->first_context_data()->qualifiers().using_batching) {
      cout << "Given encryption parameters do not support batching." << endl;
      return;
    }

    cout << "Generating Galois keys: ";
    time_start = chrono::high_resolution_clock::now();
    gal_keys = keygen.galois_keys();
    time_end = chrono::high_resolution_clock::now();
    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]" << endl;
  }

  Encryptor encryptor(context, public_key);
  Decryptor decryptor(context, secret_key);
  Evaluator evaluator(context);
  CKKSEncoder ckks_encoder(context);

  chrono::microseconds time_encode_sum(0);
  chrono::microseconds time_decode_sum(0);
  chrono::microseconds time_encrypt_sum(0);
  chrono::microseconds time_decrypt_sum(0);
  chrono::microseconds time_add_sum(0);
  chrono::microseconds time_multiply_sum(0);
  chrono::microseconds time_multiply_plain_sum(0);
  chrono::microseconds time_square_sum(0);
  chrono::microseconds time_relinearize_sum(0);
  chrono::microseconds time_rescale_sum(0);
  chrono::microseconds time_rotate_one_step_sum(0);
  chrono::microseconds time_rotate_random_sum(0);
  chrono::microseconds time_conjugate_sum(0);

  /*
  How many times to run the test?
  */
  long long count = 10;

  /*
  Populate a vector of floating-point values to batch.
  */
  vector<double> pod_vector;
  random_device rd;
  for (size_t i = 0; i < ckks_encoder.slot_count(); i++) {
    pod_vector.push_back(1.001 * static_cast<double>(i));
  }

  cout << "Running tests ";
  for (long long i = 0; i < count; i++) {
    /*
    [Encoding]
    For scale we use the square root of the last coeff_modulus prime
    from parms.
    */
    Plaintext plain(parms.poly_modulus_degree() * parms.coeff_modulus().size(),
                    0);
    /*
     */
    double scale =
        sqrt(static_cast<double>(parms.coeff_modulus().back().value()));
    time_start = chrono::high_resolution_clock::now();
    ckks_encoder.encode(pod_vector, scale, plain);
    time_end = chrono::high_resolution_clock::now();
    time_encode_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Decoding]
    */
    vector<double> pod_vector2(ckks_encoder.slot_count());
    time_start = chrono::high_resolution_clock::now();
    ckks_encoder.decode(plain, pod_vector2);
    time_end = chrono::high_resolution_clock::now();
    time_decode_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Encryption]
    */
    Ciphertext encrypted(context);
    time_start = chrono::high_resolution_clock::now();
    encryptor.encrypt(plain, encrypted);
    time_end = chrono::high_resolution_clock::now();
    time_encrypt_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Decryption]
    */
    Plaintext plain2(poly_modulus_degree, 0);
    time_start = chrono::high_resolution_clock::now();
    decryptor.decrypt(encrypted, plain2);
    time_end = chrono::high_resolution_clock::now();
    time_decrypt_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Add]
    */
    Ciphertext encrypted1(context);
    ckks_encoder.encode(i + 1, plain);
    encryptor.encrypt(plain, encrypted1);
    Ciphertext encrypted2(context);
    ckks_encoder.encode(i + 1, plain2);
    encryptor.encrypt(plain2, encrypted2);
    time_start = chrono::high_resolution_clock::now();
    evaluator.add_inplace(encrypted1, encrypted1);
    evaluator.add_inplace(encrypted2, encrypted2);
    evaluator.add_inplace(encrypted1, encrypted2);
    time_end = chrono::high_resolution_clock::now();
    time_add_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Multiply]
    */
    encrypted1.reserve(3);
    time_start = chrono::high_resolution_clock::now();
    evaluator.multiply_inplace(encrypted1, encrypted2);
    time_end = chrono::high_resolution_clock::now();
    time_multiply_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Multiply Plain]
    */
    time_start = chrono::high_resolution_clock::now();
    evaluator.multiply_plain_inplace(encrypted2, plain);
    time_end = chrono::high_resolution_clock::now();
    time_multiply_plain_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    /*
    [Square]
    */
    time_start = chrono::high_resolution_clock::now();
    evaluator.square_inplace(encrypted2);
    time_end = chrono::high_resolution_clock::now();
    time_square_sum +=
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);

    if (context->using_keyswitching()) {
      // /*
      // [Relinearize]
      // */
      // time_start = chrono::high_resolution_clock::now();
      // evaluator.relinearize_inplace(encrypted1, relin_keys);
      // time_end = chrono::high_resolution_clock::now();
      // time_relinearize_sum += chrono::duration_cast<
      //     chrono::microseconds>(time_end - time_start);

      /*
      [Rescale]
      */
      time_start = chrono::high_resolution_clock::now();
      evaluator.rescale_to_next_inplace(encrypted1);
      time_end = chrono::high_resolution_clock::now();
      time_rescale_sum +=
          chrono::duration_cast<chrono::microseconds>(time_end - time_start);

      // /*
      // [Rotate Vector]
      // */
      // time_start = chrono::high_resolution_clock::now();
      // evaluator.rotate_vector_inplace(encrypted, 1, gal_keys);
      // evaluator.rotate_vector_inplace(encrypted, -1, gal_keys);
      // time_end = chrono::high_resolution_clock::now();
      // time_rotate_one_step_sum += chrono::duration_cast<
      //     chrono::microseconds>(time_end - time_start);

      // /*
      // [Rotate Vector Random]
      // */
      // int random_rotation = static_cast<int>(rd() %
      // ckks_encoder.slot_count()); time_start =
      // chrono::high_resolution_clock::now();
      // evaluator.rotate_vector_inplace(encrypted, random_rotation, gal_keys);
      // time_end = chrono::high_resolution_clock::now();
      // time_rotate_random_sum += chrono::duration_cast<
      //     chrono::microseconds>(time_end - time_start);

      // /*
      // [Complex Conjugate]
      // */
      // time_start = chrono::high_resolution_clock::now();
      // evaluator.complex_conjugate_inplace(encrypted, gal_keys);
      // time_end = chrono::high_resolution_clock::now();
      // time_conjugate_sum += chrono::duration_cast<
      //     chrono::microseconds>(time_end - time_start);
    }

    /*
    Print a dot to indicate progress.
    */
    cout << ".";
    cout.flush();
  }

  cout << " Done" << endl << endl;
  cout.flush();

  // auto avg_encode = time_encode_sum.count() / count;
  // auto avg_decode = time_decode_sum.count() / count;
  // auto avg_encrypt = time_encrypt_sum.count() / count;
  // auto avg_decrypt = time_decrypt_sum.count() / count;
  // auto avg_add = time_add_sum.count() / (3 * count);
  // auto avg_multiply = time_multiply_sum.count() / count;
  // auto avg_multiply_plain = time_multiply_plain_sum.count() / count;
  // auto avg_square = time_square_sum.count() / count;
  // auto avg_relinearize = time_relinearize_sum.count() / count;
  auto avg_rescale = time_rescale_sum.count() / count;
  // auto avg_rotate_one_step = time_rotate_one_step_sum.count() / (2 * count);
  // auto avg_rotate_random = time_rotate_random_sum.count() / count;
  // auto avg_conjugate = time_conjugate_sum.count() / count;

  // cout << "Average encode: " << avg_encode << " microseconds" << endl;
  // cout << "Average decode: " << avg_decode << " microseconds" << endl;
  // cout << "Average encrypt: " << avg_encrypt << " microseconds" << endl;
  // cout << "Average decrypt: " << avg_decrypt << " microseconds" << endl;
  // cout << "Average add: " << avg_add << " microseconds" << endl;
  // cout << "Average multiply: " << avg_multiply << " microseconds" << endl;
  // cout << "Average multiply plain: " << avg_multiply_plain << " microseconds"
  // << endl; cout << "Average square: " << avg_square << " microseconds" <<
  // endl;
  if (context->using_keyswitching()) {
    // cout << "Average relinearize: " << avg_relinearize << " microseconds" <<
    // endl;
    cout << "Average rescale: " << avg_rescale << " microseconds" << endl;
    // cout << "Average rotate vector one step: " << avg_rotate_one_step <<
    //     " microseconds" << endl;
    // cout << "Average rotate vector random: " << avg_rotate_random << "
    // microseconds" << endl; cout << "Average complex conjugate: " <<
    // avg_conjugate << " microseconds" << endl;
  }
  cout.flush();
}

// https://github.com/microsoft/SEAL/blob/master/native/examples/6_performance.cpp
// Fixed by @musaprg
void example_ckks_performance_default() {
  print_example_banner(
      "CKKS Performance Test with Degrees: 4096, 8192, and 16384");

  // It is not recommended to use BFVDefault primes in CKKS. However, for
  // performance test, BFVDefault primes are good enough.
  EncryptionParameters parms(scheme_type::CKKS);
  size_t poly_modulus_degree = 4096;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
  ckks_performance_test(SEALContext::Create(parms));

  cout << endl;
  poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
  ckks_performance_test(SEALContext::Create(parms));

  cout << endl;
  poly_modulus_degree = 16384;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
  ckks_performance_test(SEALContext::Create(parms));

  cout << endl;
  poly_modulus_degree = 32768;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
  ckks_performance_test(SEALContext::Create(parms));
}

void sample() {
  EncryptionParameters parms(scheme_type::CKKS);

  size_t poly_modulus_degree = 8192;
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(
      CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

  double scale = pow(2.0, 40);

  auto context = SEALContext::Create(parms);
    print_parameters(context);
    cout << endl;

    KeyGenerator keygen(context);
    auto public_key = keygen.public_key();
    auto secret_key = keygen.secret_key();
    auto relin_keys = keygen.relin_keys();
    Encryptor encryptor(context, public_key);

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

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

    Plaintext x_plain;
    print_line(__LINE__);
    cout << "Encode input vectors." << endl;
    encoder.encode(input, scale, x_plain);
    Ciphertext x1_encrypted;
    encryptor.encrypt(x_plain, x1_encrypted);

    auto x1data = x1_encrypted.data();
    using uint64_t_p = uint64_t*;
    uint64_t_p pointer_to_data = static_cast<uint64_t_p>(x1data);

    // check if it can be converted to pointer
    cout << *x1data << endl;
    cout << *pointer_to_data << endl;
    // ok

    cout << *(x1data + 1) << endl;
    cout << *(pointer_to_data + 1) << endl;

    {
      auto array = cuda::make_unique<uint64_t[]>(3);
      //        cout << type_name<decltype(array)>() << endl;
      proxy();
    }

    auto context_data_ptr = context->get_context_data(x1_encrypted.parms_id());
    auto &context_data = *context_data_ptr;
    auto &next_context_data = *context_data.next_context_data();
    auto &next_parms = next_context_data.parms();

    // smallmodulus -> uint64_t
    // it seems not to cast simply.
    auto &next_coeff_modulus = next_parms.coeff_modulus();
    vector<uint64_t> coeff_modulus;
    coeff_modulus.reserve(next_coeff_modulus.size());
    for (auto &&v : next_coeff_modulus) {
      coeff_modulus.push_back(v.value());
    }
    //    print_vector(coeff_modulus);
}

int main() {
  //    example_ckks_basics();

  // sample();

  example_ckks_performance_default();

  return 0;
}
