#!/bin/bash

set -eu

ROOTPATH=$(git rev-parse --show-toplevel)
LOGPATH=$ROOTPATH/debug_logs
FNAME=$(date "+%Y%m%d-%H%M%S").log

mkdir -p $LOGPATH

$ROOTPATH/bin/sample &> $LOGPATH/$FNAME
