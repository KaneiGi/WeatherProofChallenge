CUDA_VISIBLE_DEVICES=0 python demo/demo.py --config-file configs/weatherproof/swin/oneformer_swin_large_bs16_100ep.yaml \
  --input datasets/WeatherProofTest \
  --output datasets/WeatherProofTestOut \
  --task semantic \
  --confidence-threshold 0.5 \
  --opts MODEL.IS_TRAIN False MODEL.IS_DEMO True MODEL.WEIGHTS outputs/weatherproof_swin_large_extra_30/model_0007499.pth
