'''
conda activate UDAVSS
cd xxxxPATHxxxx/video_udass/VIDEO
python ./tps/scripts_ablation/train_DAVSS_DSF_cd_ablation_24_former.py --cfg ./tps/scripts_ablation/configs/tps_syn2city.yml


'''



import argparse
import os
import os.path as osp
import pprint
import random
import warnings
import numpy as np
import yaml
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
import sys
sys.path.append('xxxxPATHxxxx/video_udass/VIDEO')

from tps.domain_adaptation_ablation.config_v100 import cfg, cfg_from_file

from torch.utils import data
# from tps.model.accel_deeplabv2_dsp_cd_DFF_gaijin_former import get_accel_deeplab_v2
from tps.model.get_accel_deeplab_v2_tps_former import get_accel_deeplab_v2_tps
from tps.model.mix_transformer2_tps import DAFormer_tps

from tps.dataset.Viper import ViperDataSet
from tps.dataset.SynthiaSeq import SynthiaSeqDataSet
from tps.dataset.CityscapesSeq import CityscapesSeqDataSet
from tps.domain_adaptation_ablation.train_video_DAVSS_DSF_cd_ablation_24_former import train_domain_adaptation

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, default=None,
                        help='optional config file', )
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", type=bool, default=True,
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    parser.add_argument("--exp-suffix", type=str, default=None,
                        help="optional experiment suffix")
    return parser.parse_args()


def main():
    # LOAD ARGS
    args = get_arguments()
    print('Called with args:')
    print(args)

    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)
    # auto-generate exp name if not specified
    if cfg.EXP_NAME == '':
        cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}_DAVSS_DSF_cd_ablation_24_former'

    if args.exp_suffix:
        cfg.EXP_NAME += f'_{args.exp_suffix}'
    # auto-generate snapshot path if not specified
    if cfg.TRAIN.SNAPSHOT_DIR_DSP_CD == '':
        cfg.TRAIN.SNAPSHOT_DIR_DSP_CD = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR_DSP_CD, exist_ok=True)
    else:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR_DSP_CD, exist_ok=True)
    # tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR_CD == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR_CD = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
            print('cfg.TRAIN.TENSORBOARD_LOGDIR_CD', cfg.TRAIN.TENSORBOARD_LOGDIR_CD)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR_CD, exist_ok=True)
        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR_CD = ''
    print('Using config:')
    pprint.pprint(cfg)

    # INIT
    _init_fn = None
    if not args.random_train:
        torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
        torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
        np.random.seed(cfg.TRAIN.RANDOM_SEED)
        random.seed(cfg.TRAIN.RANDOM_SEED)

        def _init_fn(worker_id):
            np.random.seed(cfg.TRAIN.RANDOM_SEED + worker_id)

    if os.environ.get('ADVENT_DRY_RUN', '0') == '1':
        return

    if cfg.SOURCE == 'Viper':
        RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_viper2city_pretrained.pth'
    elif cfg.SOURCE == 'SynthiaSeq':
        RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'


    # LOAD SEGMENTATION NET
    assert osp.exists(RESTORE_FROM), f'Missing init model {RESTORE_FROM}'

    if cfg.TRAIN.MODEL == 'ACCEL_DeepLabv2':
        model_tps = get_accel_deeplab_v2_tps(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(RESTORE_FROM)
        model_tps.load_state_dict(saved_state_dict)
    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    daformer_model = DAFormer_tps(device = device, num_classes = cfg.NUM_CLASSES)
    daformer_model.model.init_weights()
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'
    # RESTORE_FROM = '/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models/tps_syn2city_pretrained.pth'

    print('Model loaded')

    # DATALOADERS
    if cfg.SOURCE == 'Viper':
        source_dataset = ViperDataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                     list_path=cfg.DATA_LIST_SOURCE,
                                     set=cfg.TRAIN.SET_SOURCE,
                                     max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                     crop_size=cfg.TRAIN.INPUT_SIZE_SOURCE,
                                     mean=cfg.TRAIN.IMG_MEAN)
    
    elif cfg.SOURCE == 'SynthiaSeq':
        INPUT_SIZE_SOURCE = (1280, 760)
        source_dataset = SynthiaSeqDataSet(root=cfg.DATA_DIRECTORY_SOURCE,
                                     list_path=cfg.DATA_LIST_SOURCE,
                                     set=cfg.TRAIN.SET_SOURCE,
                                     max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_SOURCE,
                                     crop_size=INPUT_SIZE_SOURCE,
                                     mean=cfg.TRAIN.IMG_MEAN)
    source_loader = data.DataLoader(source_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_SOURCE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    target_dataset = CityscapesSeqDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=cfg.TRAIN.MAX_ITERS * cfg.TRAIN.BATCH_SIZE_TARGET,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN,
                                       interval=cfg.TRAIN.INTERVAL)
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)

    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR_DSP_CD, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    train_domain_adaptation(daformer_model, model_tps, source_loader, target_loader, cfg, device)

if __name__ == '__main__':
    main()


