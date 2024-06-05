#!/bin/bash

python tools/test.py ../configs/depth_anything/depth_anything_large_setrmla_1xb4_160k_weather_proof_cdv4train_whole_896x896.py \
 checkpoints/2-setrmla.pth --work-dir work_dir/setrmla_whole --out work_dir/setrmla_whole --show-dir work_dir/setrmla_whole