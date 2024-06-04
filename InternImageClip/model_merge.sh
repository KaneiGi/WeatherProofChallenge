#!/usr/bin/env bash

RGB_PATH=$1
LABEL_PATH=$2
OUT_LABEL_PATH=$3
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/model_merge.py $RGB_PATH $LABEL_PATH $OUT_LABEL_PATH