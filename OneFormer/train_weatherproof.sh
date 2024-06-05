export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_net.py --dist-url 'tcp://127.0.0.1:50163' \
    --num-gpus 8 \
    --config-file configs/weatherproof/swin/oneformer_swin_large_bs16_100ep.yaml \
    OUTPUT_DIR work_dirs/weatherproof_swin_large_extra_30 MODEL.WEIGHTS weights/150_16_swin_l_oneformer_coco_100ep.pth
