//
// Created by kotaro on 2020/01/20.
//

#include "performance_evaluator.h"

void bench()
{
    print_example_banner(
      "CKKS Performance Test with Degrees: 4096, 8192, 16384 and 32768");

    // It is not recommended to use BFVDefault primes in CKKS. However, for
    // performance test, BFVDefault primes are good enough.
    seal::EncryptionParameters parms(seal::scheme_type::CKKS);
    size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
      seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    bench_cpu(seal::SEALContext::Create(parms));
    bench_gpu(seal::SEALContext::Create(parms));

    cout << endl;
    poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
      seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    bench_cpu(seal::SEALContext::Create(parms));
    bench_gpu(seal::SEALContext::Create(parms));

    cout << endl;
    poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
      seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    bench_cpu(seal::SEALContext::Create(parms));
    bench_gpu(seal::SEALContext::Create(parms));

    cout << endl;
    poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
      seal::CoeffModulus::BFVDefault(poly_modulus_degree));
    bench_cpu(seal::SEALContext::Create(parms));
    bench_gpu(seal::SEALContext::Create(parms));
}

// https://github.com/microsoft/SEAL/blob/master/native/examples/6_performance.cpp
// Fixed by @musaprg
void bench_cpu(shared_ptr<seal::SEALContext> context)
{
    chrono::steady_clock::time_point time_start, time_end;

    print_parameters(context);
    cout << endl;

    auto &parms = context->first_context_data()->parms();
    size_t poly_modulus_degree = parms.poly_modulus_degree();

    cout << "Generating secret/public keys: ";
    seal::KeyGenerator keygen(context);
    cout << "Done" << endl;

    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key();

    seal::RelinKeys relin_keys;
    chrono::microseconds time_diff;

    cout << "Generating relinearization keys: ";
    time_start = chrono::steady_clock::now();
    relin_keys = keygen.relin_keys();
    time_end = chrono::steady_clock::now();
    time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]" << endl;

    seal::Encryptor encryptor(context, public_key);
    seal::Decryptor decryptor(context, secret_key);
    seal::Evaluator evaluator(context);
    seal::CKKSEncoder ckks_encoder(context);

    chrono::microseconds time_multiply_sum(0);
    chrono::microseconds time_relinearize_sum(0);
    chrono::microseconds time_rescale_sum(0);

    /*
    How many times to run the test?
    */
    long long count = 10;

    /*
    Populate a vector of floating-point values to batch.
    */
    vector<double> pod_vector;
    random_device rd;
    for (size_t i = 0; i < ckks_encoder.slot_count(); i++)
    {
        pod_vector.push_back(1.001 * static_cast<double>(i));
    }

    cout << "Running tests ";
    for (long long i = 0; i < count; i++)
    {
        /*
        [Encoding]
        For scale we use the square root of the last coeff_modulus prime
        from parms.
        */
        seal::Plaintext plain(
          parms.poly_modulus_degree() * parms.coeff_modulus().size(), 0);
        /*
         */
        double scale =
          sqrt(static_cast<double>(parms.coeff_modulus().back().value()));
        ckks_encoder.encode(pod_vector, scale, plain);

        /*
        [Encryption]
        */
        seal::Ciphertext encrypted(context);
        encryptor.encrypt(plain, encrypted);

        /*
        [Add]
        */
        seal::Ciphertext encrypted1(context);
        ckks_encoder.encode(i + 1, plain);
        encryptor.encrypt(plain, encrypted1);
        seal::Ciphertext encrypted2(context);
        ckks_encoder.encode(i + 1, plain);
        encryptor.encrypt(plain, encrypted2);
        evaluator.add_inplace(encrypted1, encrypted1);
        evaluator.add_inplace(encrypted2, encrypted2);
        evaluator.add_inplace(encrypted1, encrypted2);

        /*
        [Multiply]
        */
        encrypted1.reserve(3);
        time_start = chrono::steady_clock::now();
        evaluator.multiply_inplace(encrypted1, encrypted2);
        time_end = chrono::steady_clock::now();
        time_multiply_sum +=
          chrono::duration_cast<chrono::microseconds>(time_end - time_start);

        /*
        [Relinearize]
        */
        time_start = chrono::steady_clock::now();
        evaluator.relinearize_inplace(encrypted1, relin_keys);
        time_end = chrono::steady_clock::now();
        time_relinearize_sum +=
          chrono::duration_cast<chrono::microseconds>(time_end - time_start);

        /*
        [Rescale]
        */
        time_start = chrono::steady_clock::now();
        evaluator.rescale_to_next_inplace(encrypted1);
        time_end = chrono::steady_clock::now();
        time_rescale_sum +=
          chrono::duration_cast<chrono::microseconds>(time_end - time_start);

        /*
        Print a dot to indicate progress.
        */
        cout << ".";
        cout.flush();
    }

    cout << " Done" << endl << endl;
    cout.flush();

    auto avg_rescale = time_rescale_sum.count() / count;
    auto avg_multiply = time_multiply_sum.count() / count;
    auto avg_relinearize = time_relinearize_sum.count() / count;

    cout << "[CPU] Average rescale: " << avg_rescale << " microseconds" << endl;

    auto sum_of_multiplication_time =
      avg_multiply + avg_relinearize + avg_rescale;
    cout << "[CPU] Ratio of multiply: "
         << static_cast<double>(avg_multiply) * 100 / sum_of_multiplication_time
         << endl;
    cout << "[CPU] Ratio of relinearization: "
         << static_cast<double>(avg_relinearize) * 100 /
              sum_of_multiplication_time
         << endl;
    cout << "[CPU] Ratio of rescale: "
         << static_cast<double>(avg_rescale) * 100 / sum_of_multiplication_time
         << endl;
}

// https://github.com/microsoft/SEAL/blob/master/native/examples/6_performance.cpp
// Fixed by @musaprg
void bench_gpu(shared_ptr<seal::SEALContext> context)
{
    chrono::steady_clock::time_point time_start, time_end;

    print_parameters(context);
    cout << endl;

    auto &parms = context->first_context_data()->parms();
    size_t poly_modulus_degree = parms.poly_modulus_degree();

    cout << "Generating secret/public keys: ";
    seal::KeyGenerator keygen(context);
    cout << "Done" << endl;

    auto secret_key = keygen.secret_key();
    auto public_key = keygen.public_key();

    seal::RelinKeys relin_keys;
    chrono::microseconds time_diff;

    cout << "Generating relinearization keys: ";
    time_start = chrono::steady_clock::now();
    relin_keys = keygen.relin_keys();
    time_end = chrono::steady_clock::now();
    time_diff =
      chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "Done [" << time_diff.count() << " microseconds]" << endl;

    seal::Encryptor encryptor(context, public_key);
    seal::Decryptor decryptor(context, secret_key);
    seal::Evaluator evaluator(context);
    seal::CKKSEncoder ckks_encoder(context);

    chrono::microseconds time_multiply_sum(0);
    chrono::microseconds time_relinearize_sum(0);
    double time_rescale_sum(0);
    double time_ntt_sum(0);
    double time_inverse_ntt_sum(0);
    double time_data_transmission_sum(0);

    /*
    How many times to run the test?
    */
    long long count = 10;

    /*
    Populate a vector of floating-point values to batch.
    */
    vector<double> pod_vector;
    random_device rd;
    for (size_t i = 0; i < ckks_encoder.slot_count(); i++)
    {
        pod_vector.push_back(1.001 * static_cast<double>(i));
    }

    cout << "Running tests ";
    for (long long i = 0; i < count; i++)
    {
        /*
        [Encoding]
        For scale we use the square root of the last coeff_modulus prime
        from parms.
        */
        seal::Plaintext plain(
          parms.poly_modulus_degree() * parms.coeff_modulus().size(), 0);
        /*
         */
        double scale =
          sqrt(static_cast<double>(parms.coeff_modulus().back().value()));
        ckks_encoder.encode(pod_vector, scale, plain);

        /*
        [Encryption]
        */
        seal::Ciphertext encrypted(context);
        encryptor.encrypt(plain, encrypted);

        /*
        [Add]
        */
        seal::Ciphertext encrypted1(context);
        ckks_encoder.encode(i + 1, plain);
        encryptor.encrypt(plain, encrypted1);
        seal::Ciphertext encrypted2(context);
        ckks_encoder.encode(i + 1, plain);
        encryptor.encrypt(plain, encrypted2);
        evaluator.add_inplace(encrypted1, encrypted1);
        evaluator.add_inplace(encrypted2, encrypted2);
        evaluator.add_inplace(encrypted1, encrypted2);

        /*
        [Multiply]
        */
        encrypted1.reserve(3);
        time_start = chrono::steady_clock::now();
        evaluator.multiply_inplace(encrypted1, encrypted2);
        time_end = chrono::steady_clock::now();
        time_multiply_sum +=
          chrono::duration_cast<chrono::microseconds>(time_end - time_start);

        /*
        [Relinearize]
        */
        time_start = chrono::steady_clock::now();
        evaluator.relinearize_inplace(encrypted1, relin_keys);
        time_end = chrono::steady_clock::now();
        time_relinearize_sum +=
          chrono::duration_cast<chrono::microseconds>(time_end - time_start);

        /*
        [Rescale]
        */
        seal::Ciphertext destination = encrypted1;
        CudaContextData context_cu =
          get_cuda_context_data(context, encrypted1, destination);
        auto encrypted1_cu = get_cuciphertext_from_ciphertext(encrypted1);
        auto destination_cu = get_cuciphertext_from_ciphertext(encrypted1);
        // time_start = chrono::steady_clock::now();
        double rescale_time, ntt_time, inverse_ntt_time, data_transmission_time;
        auto elapsed_time =
          rescale_to_next(encrypted1_cu, destination_cu, context_cu);
        // time_end = chrono::steady_clock::now();
        // time_rescale_sum += chrono::duration_cast<chrono::microseconds>(
        //   rescale_time + ntt_time + inverse_ntt_time);
        time_rescale_sum += elapsed_time.rescale_time;
        time_ntt_sum += elapsed_time.ntt_time;
        time_inverse_ntt_sum += elapsed_time.inverse_ntt_time;
        time_data_transmission_sum += elapsed_time.data_transmission_time;

        /*
        Print a dot to indicate progress.
        */
        cout << ".";
        cout.flush();
    }

    cout << " Done" << endl << endl;
    cout.flush();

    auto avg_rescale = time_rescale_sum / count;
    auto avg_ntt = time_ntt_sum / count;
    auto avg_inverse_ntt = time_inverse_ntt_sum / count;
    auto avg_data_transmission = time_data_transmission_sum / count;
    auto avg_whole_rescale = avg_rescale + avg_ntt + avg_inverse_ntt;

    cout << "[GPU] Average rescale: " << avg_rescale << " microseconds" << endl;
    cout << "[GPU] Average ntt: " << avg_ntt << " microseconds" << endl;
    cout << "[GPU] Average intt: " << avg_inverse_ntt << " microseconds"
         << endl;
    cout << "[GPU] Average rescale(whole): " << avg_whole_rescale
         << " microseconds" << endl;
    cout << "[GPU] Average data_transmission: " << avg_data_transmission
         << " microseconds" << endl;
    cout << "[GPU] Average rescale(whole + data_transmission): "
         << avg_whole_rescale + avg_data_transmission << " microseconds"
         << endl;
}

int main()
{
    bench();
    return 0;
}