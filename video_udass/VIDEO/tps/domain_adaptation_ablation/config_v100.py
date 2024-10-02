import os.path as osp
import numpy as np
from easydict import EasyDict
from tps.utils import project_root, project_root_ADVENT
from ADVENT.advent.utils.serialization import yaml_load

cfg = EasyDict()


cfg.N_SEGMENTS = 4
cfg.tar_loss_weight = 0.6

cfg.tensorboard = True
cfg.generate_tcsm_synthia_path = ''
cfg.prototype_path = str(project_root / 'experiments_result/prototype')
cfg.train_thred = 0.0
cfg.proto_temperature = 1.0
cfg.regular_type = 'MRKLD' # 'MRENT|MRKLD'
cfg.regular_w = 0.1
cfg.ema_w = 0.5
cfg.lamda = 1.0
cfg.loss_cts_W = 0.5
cfg.proto_consistW = 10.0
cfg.pixel_weight = "threshold_uniform"
cfg.dsp_hard_loss = True # False || True

cfg.NUM_WORKERS = 4
cfg.EXP_NAME = ''
cfg.EXP_ROOT = project_root / 'experiments_result'
cfg.project = project_root
cfg.EXP_ROOT_SNAPSHOT = osp.join(cfg.EXP_ROOT, 'snapshots')
cfg.EXP_ROOT_LOGS = osp.join(cfg.EXP_ROOT, 'logs')

cfg.TENSORBOARD_DIR = ''
cfg.proto_resume = False
cfg.alpha_mix = 0.5

cfg.generate_threshold_path = ''
cfg.ignore_index = 250

cfg.TRAIN = EasyDict()
cfg.TRAIN.BATCH_SIZE = 1

cfg.TRAIN.SET_SOURCE = 'all'
cfg.TRAIN.SET_TARGET = 'train'
cfg.TRAIN.BATCH_SIZE_SOURCE = 1
cfg.TRAIN.BATCH_SIZE_TARGET = 1
cfg.TRAIN.IGNORE_LABEL = 255
cfg.TRAIN.INPUT_SIZE_SOURCE = (1280, 720)
# cfg.TRAIN.INPUT_SIZE_SOURCE = (1024, 512)
cfg.TRAIN.INPUT_SIZE_TARGET = (1024, 512)
cfg.TRAIN.INFO_SOURCE = ''
cfg.TRAIN.MULTI_LEVEL = True

cfg.TRAIN.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TRAIN.LEARNING_RATE = 2.5e-4
cfg.TRAIN.MOMENTUM = 0.9
cfg.TRAIN.WEIGHT_DECAY = 0.0005
cfg.TRAIN.POWER = 0.9
cfg.TRAIN.LAMBDA_SEG_MAIN = 1.0
cfg.TRAIN.LAMBDA_SEG_AUX = 0.1
cfg.TRAIN.MAX_ITERS = 250000
cfg.TRAIN.EARLY_STOP = 120000
cfg.TRAIN.SAVE_PRED_EVERY = 1000 # 100
cfg.TRAIN.SNAPSHOT_DIR = ''
cfg.TRAIN.RANDOM_SEED = 1234
cfg.TRAIN.TENSORBOARD_VIZRATE = 1000 #100
cfg.TRAIN.LAMBDA_T = 1.0

# TPS
cfg.TRAIN.DA_METHOD = 'SourceOnly'
cfg.NUM_CLASSES = 15
cfg.SOURCE = 'Viper'
cfg.TARGET = 'CityscapesSeq'
cfg.TRAIN.INFO_TARGET = str(project_root / 'tps/dataset/CityscapesSeq_list/info_Viper.json')
cfg.DATA_LIST_SOURCE = str(project_root / 'tps/dataset/Viper_list/{}.txt')
cfg.DATA_LIST_TARGET = str(project_root / 'tps/dataset/CityscapesSeq_list/{}.txt')
cfg.DATA_DIRECTORY_SOURCE = ''
cfg.DATA_DIRECTORY_TARGET = 'data/Cityscapes'
cfg.TRAIN.MODEL = 'ACCEL_DeepLabv2'
cfg.TRAIN.flow_path_src = ''
cfg.TRAIN.flow_path = 'data/estimated_optical_flow_cityscapes_seq_train'
cfg.TRAIN.INTERVAL = 1
cfg.TRAIN.THRESHOLD = 0.0
cfg.TRAIN.SCALING_RATIO = (0.8, 1.2)
cfg.TRAIN.RESTORE_FROM = 'pretrained_models/DeepLab_resnet_pretrained_imagenet.pth'
cfg.TRAIN.RESTORE_FROM_viper = 'pretrained_models/tps_viper2city_pretrained.pth'
cfg.TRAIN.RESTORE_FROM_synthia = 'pretrained_models/tps_syn2city_pretrained.pth'



# TEST CONFIGS
cfg.TEST = EasyDict()
cfg.TEST.MODE = 'video_single'  # {'video_single', 'video_best', 'video_best_tcsm'}
cfg.TEST.MODEL = ('ACCEL_DeepLabv2',)
cfg.TEST.RESTORE_FROM_prototype = ('',)
cfg.TEST.MODEL_WEIGHT = (1.0,)
cfg.TEST.MULTI_LEVEL = (True,)
cfg.TEST.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
cfg.TEST.SNAPSHOT_DIR = ('',)
cfg.TEST.RESTORE_FROM = ('',)
cfg.TEST.DAVSS_DSF_cd_ablation_31 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models',) # VIPER2SYN CNN
cfg.TEST.DAVSS_DSF_cd_ablation_24 = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models',) # SYN2SYN CNN
cfg.TEST.DAVSS_DSF_cd_ablation_31_former = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models',) # VIPER2SYN VIT
cfg.TEST.DAVSS_DSF_cd_ablation_24_former = ('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/pretrained_models',) # SYN2SYN VIT

cfg.TEST.SNAPSHOT_STEP = 200  # used in 'best' mode
cfg.TEST.SNAPSHOT_START_ITER = 200  # used in 'best' mode
cfg.TEST.SNAPSHOT_MAXITER = 40000  # used in 'best' mode
cfg.TEST.SET_TARGET = 'val'
cfg.TEST.BATCH_SIZE_TARGET = 1
cfg.TEST.INPUT_SIZE_TARGET = (1024, 512)
cfg.TEST.OUTPUT_SIZE_TARGET = (2048, 1024)
cfg.TEST.INFO_TARGET = str(project_root / 'tps/dataset/cityscapes_list/info.json')
cfg.TEST.WAIT_MODEL = True
cfg.TEST.INTERVAL = 1
cfg.TEST.flow_path = 'data/estimated_optical_flow_cityscapes_seq_val'



# calculate prototype
cfg.proto_momentum = 0.1 # 0.9
def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not EasyDict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        # if not b.has_key(k):
        if k not in b:
            raise KeyError(f'{k} is not a valid config key')

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(f'Type mismatch ({type(b[k])} vs. {type(v)}) '
                                 f'for config key: {k}')

        # recursively merge dicts
        if type(v) is EasyDict:
            try:
                _merge_a_into_b(a[k], b[k])
            except Exception:
                print(f'Error under config key: {k}')
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options.
    """
    yaml_cfg = EasyDict(yaml_load(filename))
    _merge_a_into_b(yaml_cfg, cfg)
