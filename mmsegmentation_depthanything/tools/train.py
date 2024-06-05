# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', help='train config file path',
                        # default='../configs/depth_anything/depth_anything_large_mask2former_16xb1_160k_weather_proof_512x512.py')
                        # default='../configs/depth_anything/depth_anything_large_mask2former_16xb1_160k_weather_proof_512x512.py')
                        # default='../configs/depth_anything/depth_anything_large_segmenter_1xb12_160k_weather_proof_add_clean_448x448.py')
                        # default='../configs/depth_anything/depth_anything_large_setrmla_1xb12_160k_weather_proof_add_clean_448x448.py')
                        default='../configs/depth_anything/depth_anything_large_setrmla_1xb4_160k_weather_proof_cdv4train_test_896x896.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models',
                        # default='../work_dirs/depth_anything_large_segmenter_1xb12_160k_weather_proof_add_clean_448x448')
                        # default='../work_dirs/depth_anything_large_setrmla_1xb12_160k_weather_proof_add_clean_448x448')
                        default = '../work_dirs/setrmla_clip_cvd_896_whole_test20_finetune_finetune')

    parser.add_argument(
        '--resume',
        action='store_false',
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--load_from',
        default='../work_dirs/setrmla_clip_cvd_896_whole_test20_finetune/best_mIoU_iter_13300.pth'
    )

    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume
    ## add load_from
    cfg.load_from = args.load_from ###

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
