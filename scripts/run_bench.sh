#!/bin/bash

set -eu

ROOTPATH=$(git rev-parse --show-toplevel)
LOGPATH=$ROOTPATH/logs
FNAME=$(date "+%Y%m%d-%H%M%S").log

mkdir -p $LOGPATH

numactl --physcpubind=0 --membind=1 $ROOTPATH/bin/bench &> $LOGPATH/$FNAME
