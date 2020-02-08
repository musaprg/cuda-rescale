# cuda-rescale

**NOTE: This program is very ugly. I want to fix whole source codes, but there's no time to do so. I priortize my graduation, of course.**

## Prerequisities

- CMake v3.10 above
- gcc (prefer v7 above)
- CUDA 9.0 above
- Microsoft SEAL v3.3.2

## How to build

You must install Microsoft SEAL before building & executing this program.

```
cd third_party/seal/native/src
mkdir build
cd build
cmake ..
```

After installing SEAL, execute these commands below for building my sources.

```
mkdir build
cd build
cmake ..
make
```

If you want to run benchmarking program, execute

```
bin/bench
```
