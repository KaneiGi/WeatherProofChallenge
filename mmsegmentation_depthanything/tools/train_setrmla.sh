#!/bin/bash

python tools/train.py ../configs/depth_anything/depth_anything_large_setrmla_1xb4_160k_weather_proof_cdv4train_whole_896x896.py \
--work-dir work_dir/setrmla
