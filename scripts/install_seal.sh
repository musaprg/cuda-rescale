#!/bin/bash

set -eu

ROOTPATH=$(git rev-parse --show-toplevel)
THIRDPARTY=$ROOTPATH/third_party
SEALPATH=$THIRDPARTY/seal/native/src

pushd $SEALPATH
rm -rf build
mkdir build
pushd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
#cmake ..
make
sudo make install
