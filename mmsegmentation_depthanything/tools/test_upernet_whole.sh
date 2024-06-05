#!/bin/bash

python tools/test.py ../configs/depth_anything/depth_anything_large_upernet_1xb4_160k_weather_proof_cdv4train_whole_896x896.py \
 checkpoints/6-upernet_segmenter.pth --work-dir work_dir/upernet_whole --out work_dir/upernet_whole --show-dir work_dir/upernet_whole