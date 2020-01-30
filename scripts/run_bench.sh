#!/bin/bash

set -eu

ROOTPATH=$(git rev-parse --show-toplevel)

numactl --physcpubind=0 --membind=1 $ROOTPATH/bin/sample