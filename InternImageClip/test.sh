#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/test.py $CONFIG $CHECKPOINT --show-dir ./result --out clip_640_aug_all_epoch3_slide_finetune_15x.pkl
