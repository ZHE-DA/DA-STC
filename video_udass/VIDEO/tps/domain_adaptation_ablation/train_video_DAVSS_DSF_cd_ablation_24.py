import os
import sys
import random
from pathlib import Path
import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as T
from tqdm import tqdm
from ADVENT.advent.model.discriminator import get_fc_discriminator
from ADVENT.advent.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from ADVENT.advent.utils.func import loss_calc, bce_loss
from ADVENT.advent.utils.loss import entropy_loss
from ADVENT.advent.utils.func import prob_2_entropy
from ADVENT.advent.utils.viz_segmask import colorize_mask
from tps.utils.resample2d_package.resample2d import Resample2d
from tps.dsp.transformmasks_dsp_cd_xiuzheng import rand_mixer
from tps.dsp.transformmasks_dsp_cd_xiuzheng import generate_class_mask
from tps.dsp.transformmasks_dsp_cd_xiuzheng import Class_mix
from tps.dsp.transformmasks_dsp_cd_xiuzheng import Class_mix_flow
from tps.dsp.transformmasks_dsp_cd_xiuzheng import Class_mix_nolongtail
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib
