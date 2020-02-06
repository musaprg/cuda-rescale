//
// Created by kotaro on 2020/01/20.
//

#pragma once

#include "bridges.h"
#include "examples.hpp"
#include "functions.h"
#include "seal/seal.h"

using namespace std;

void bench();

void bench_cpu(shared_ptr<seal::SEALContext> context);

void bench_gpu(shared_ptr<seal::SEALContext> context);