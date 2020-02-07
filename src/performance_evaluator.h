//
// Created by kotaro on 2020/01/20.
//

#pragma once

#include <array>
#include <tuple>

#include "bridges.h"
#include "examples.hpp"
#include "functions.h"
#include "seal/seal.h"

using namespace std;

template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> make_array(Args&&... args)
{
    return std::array<T, sizeof...(Args)>{static_cast<Args&&>(args)...};
}

auto cuda_devices = make_array<int>(0, 6);

void bench();

void bench_cpu(shared_ptr<seal::SEALContext> context);

void bench_gpu(shared_ptr<seal::SEALContext> context, int cuda_device_id);