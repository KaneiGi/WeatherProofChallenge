#!/bin/bash

python tools/test.py ../configs/depth_anything/depth_anything_large_setrmla_1xb4_160k_weather_proof_cdv4train_slide_896x896.py \
 checkpoints/2-setrmla.pth --work-dir work_dir/setrmla_slide --out work_dir/setrmla_slide --show-dir work_dir/setrmla_slide