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


def train_domain_adaptation(model, model_tps, source_loader, target_loader, cfg, device):
    if cfg.TRAIN.DA_METHOD == 'SourceOnly':
        train_source_only(model, source_loader, target_loader, cfg, device)
    elif cfg.TRAIN.DA_METHOD == 'TPS':
        train_TPS(model, model_tps, source_loader, target_loader, cfg, device)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")

def train_source_only(model, source_loader, target_loader, cfg, device):
    if cfg.SOURCE == 'Viper':
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 0, 0, 142, 0, 0, 70,
                   0, 60, 100, 0, 0, 230, 119, 11, 32]
    elif cfg.SOURCE == 'SynthiaSeq':
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142]
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)

    def colorize_mask(mask):
        # mask: numpy array of the mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(palette)
        return new_mask

    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = True
    if cfg.tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.SNAPSHOT_DIR)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                                align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    source_loader_iter = enumerate(source_loader)
    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        # reset optimizers
        optimizer.zero_grad()
        # adapt LR if needed
        adjust_learning_rate(optimizer, i_iter, cfg)

        ######### Source-domain supervised training
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, src_label_kf,  _, src_img_name, _, _ = source_batch
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf, _, _ = model(
            src_img_cf.cuda(device), src_img_kf.cuda(device), src_flow, device)
        # pred_aux, pred, cf_aux, cf, kf_aux, kf, cf_feat, kf_feat
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)


        # trg_prob = F.softmax(src_pred, dim=1)
        # a = trg_prob.squeeze()
        # a = a.cpu().detach().numpy().transpose(1, 2, 0)
        # a = np.argmax(a, axis=2)
        # amax_output_col = colorize_mask(np.asarray(a, dtype=np.uint8))
        # amax_output_col.save('./a_viper_'+str(i_iter)+'.png')


        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0

        loss = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux
        loss.backward()
        
        optimizer.step()
        current_losses = {'loss_src': loss_seg_src_main,
                          'loss_src_aux': loss_seg_src_aux}
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)


def train_TPS(model, model_tps, source_loader, target_loader, cfg, device):
    # Create the model and start the training.
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = True
    lam = cfg.lamda
    if cfg.tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR_CD)
    # SEGMNETATION NETWORK
    model.train()
    model.to(device)
    model_tps.eval()
    model_tps.to(device)

    cudnn.benchmark = True
    cudnn.enabled = True

    # OPTIMIZERS
    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    # interpolate output segmaps
    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    # propagate predictions (of previous frames) forward
    warp_bilinear = Resample2d(bilinear=True)
    #
    source_loader_iter = enumerate(source_loader)
    target_loader_iter = enumerate(target_loader)

    class_features = Class_Features(device, numbers=cfg.NUM_CLASSES)
    
    if cfg.SOURCE == 'Viper':
        gta5_cls_mixer = rand_mixer("viper_seq", device)
        class_to_select = [3, 4, 5, 7, 13, 14]
    elif cfg.SOURCE == 'SynthiaSeq':
        gta5_cls_mixer = rand_mixer("synthia_seq", device)
        class_to_select = [3, 4, 5, 6, 10] # fence 3, pole 4, light 5, sign6, rider 10

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)): 
        # 断点续训
        # for i_iter in tqdm(range(5499, cfg.TRAIN.EARLY_STOP + 1)): 
        ####  optimizer  ####
        optimizer.zero_grad()
        
        ####  adjust LR  ####
        adjust_learning_rate(optimizer, i_iter, cfg)
        
        ####  load data  ####
        _, source_batch = source_loader_iter.__next__()
        src_img_cf, src_label, src_img_kf, src_label_kf, _, src_img_name, src_cf, src_kf = source_batch
        # print('src_img_name', src_img_name)
        # src_cf1 = src_cf
        _, target_batch = target_loader_iter.__next__()
        trg_img_d, trg_img_c, trg_img_b, trg_img_a, d,  _, name, frames = target_batch 
        # print('name', name)
        frames = frames.squeeze().tolist()
        
        ## patch_pasted
        if i_iter == 0:
            # # 断点续训
            # if i_iter == 5499:
            src_cf_last = src_img_cf.clone()
            src_kf_last = src_img_kf.clone()
            src_label_last = src_label.clone()
            src_label_last_kf = src_label_kf.clone()

            file_name = src_img_name[0].split('/')[-1]
            if cfg.SOURCE == 'Viper':
                frame = int(file_name.replace('.jpg', '')[-5:])
                frame1 = frame - 1
                flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
            elif cfg.SOURCE == 'SynthiaSeq':
                flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
            # print('flow_name', os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
            flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
            # print('flow_int16_x10.shape', flow_int16_x10.shape)
            src_flow_last_cd = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)


            file_name = name[0].split('/')[-1]

             # flow: d -> b
            flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[2]).zfill(6) + '_int16_x10')
            flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
            trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
            # flow: b -> a 
            file_name = file_name.replace(str(frames[0]).zfill(6), str(frames[2]).zfill(6))
            flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[3]).zfill(6) + '_int16_x10')
            flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
            trg_flow_b = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)


            with torch.no_grad():
                if i_iter < 8000:
                    trg_pred_aux, trg_pred, _, _, _, _ = model_tps(trg_img_b.cuda(device), kf = trg_img_a.cuda(device), flow = trg_flow_b, device = device)
                else:
                    trg_pred_aux, trg_pred, _, _, _, _, _, _ = model(trg_img_b.cuda(device), kf = trg_img_a.cuda(device), flow = trg_flow_b, device = device)
                
                trg_pred_512 = interp_target(trg_pred)
                trg_pred_aux_512 = interp_target(trg_pred_aux)

                trg_prob_512 = F.softmax(trg_pred_512, dim=1)
                interp_flow_512 = nn.Upsample(size=(trg_prob_512.shape[-2], trg_prob_512.shape[-1]), mode='bilinear', align_corners=True)
                interp_flow_ratio_512 = trg_prob_512.shape[-2] / trg_flow.shape[-2]
                trg_flow_warp_512 = (interp_flow_512(trg_flow) * interp_flow_ratio_512).float().cuda(device)
                trg_prob_warp = warp_bilinear(trg_prob_512, trg_flow_warp_512)


                trg_pl_logits, trg_pl_last = torch.max(trg_prob_warp, dim=1)
                trg_pl_kf_logits, trg_pl_kf_last = torch.max(trg_pred_512, dim=1)


                classes = torch.unique(trg_pl_last)
                nclasses = classes.shape[0]
                classes = (classes[torch.Tensor(np.random.choice(nclasses, 2 ,replace=False)).long()]) # int((nclasses+nclasses%2)/2)

                # constructing high confidence target clas-aware template different from source2 class
                mask_d_last = torch.zeros_like(trg_pl_last)
                mask_d_last[trg_pl_logits>0.9]=1.0
                # print('classes.float()', classes.float())
                # print('trg_pl_last', trg_pl_last)
                # print('mask_d_last', mask_d_last)

                mask_d_last = generate_class_mask(trg_pl_last.float(), classes) * mask_d_last

                mask_c_last = torch.zeros_like(trg_pl_kf_last)
                mask_c_last[trg_pl_kf_logits>0.9]=1.0
                mask_c_last = generate_class_mask(trg_pl_kf_last.float(), classes) * mask_c_last

                tar_d_last = trg_img_d.clone()
                tar_c_last = trg_img_c.clone()
                tar_dc_flow_last = trg_flow_warp_512.clone()

            continue

        ##  match
        src_cf = hist_match(src_cf, d)
        src_kf = hist_match(src_kf, d)
        ##  normalize
        src_cf = torch.flip(src_cf, [1])
        src_kf = torch.flip(src_kf, [1])
        src_cf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        src_kf -= torch.tensor(cfg.TRAIN.IMG_MEAN).view(1, 3, 1, 1)
        ##  recover
        src_img_cf = src_cf
        src_img_kf = src_kf

        ####  supervised | source  ####
        if src_label.dim() == 4:
            src_label = src_label.squeeze(-1)
            src_label_kf = src_label_kf.squeeze(-1)

        file_name = src_img_name[0].split('/')[-1]
        if cfg.SOURCE == 'Viper':
            frame = int(file_name.replace('.jpg', '')[-5:])
            frame1 = frame - 1
            flow_int16_x10_name = file_name.replace('.jpg', str(frame1).zfill(5) + '_int16_x10')
        elif cfg.SOURCE == 'SynthiaSeq':
            flow_int16_x10_name = file_name.replace('.png', '_int16_x10')
        # print('flow_name', os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        flow_int16_x10 = np.load(os.path.join(cfg.TRAIN.flow_path_src, flow_int16_x10_name + '.npy'))
        # print('flow_int16_x10.shape', flow_int16_x10.shape)
        src_flow = torch.from_numpy(flow_int16_x10 / 10.0).permute(2, 0, 1).unsqueeze(0)
        # print('src_flow.shape', src_flow.shape)
        src_pred_aux, src_pred, src_pred_cf_aux, src_pred_cf, src_pred_kf_aux, src_pred_kf, _, _ = model(src_img_cf.cuda(device), kf = src_img_kf.cuda(device), flow = src_flow, device = device)
        src_pred1=src_pred
        src_pred = interp_source(src_pred)
        loss_seg_src_main = loss_calc(src_pred, src_label, device)
        if cfg.TRAIN.MULTI_LEVEL:
            src_pred_aux = interp_source(src_pred_aux)
            loss_seg_src_aux = loss_calc(src_pred_aux, src_label, device)
        else:
            loss_seg_src_aux = 0
        loss_sou = cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux

        src_label_pro = src_label.clone().float().unsqueeze(0).cuda(device)
        if cfg.SOURCE == 'SynthiaSeq':
            src_label_pro = F.interpolate(src_label_pro, size=(81, 161), mode='nearest')
        else:
            src_label_pro = F.interpolate(src_label_pro, size=(91, 161), mode='nearest')

        objective_vectors_sou = torch.zeros([cfg.NUM_CLASSES, cfg.NUM_CLASSES]).cuda(device)
        vectors, ids = class_features.calculate_mean_vector(src_pred1, src_pred1, labels=src_label_pro)
        for t in range(len(ids)):
            objective_vectors_sou = update_objective_SingleVector(objective_vectors_sou, ids[t], vectors[t])
            # if t ==0:
            #     print('objective_vectors_sou[t]', objective_vectors_sou[t])

        ####  unsupervised | target  ####
        ##  optical flow  ##
        '''
            {d, c} or {b, a}: pair of consecutive frames extracted from the same video
        '''
        file_name = name[0].split('/')[-1]
        # flow: d -> c
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[1]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow_d = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        # flow: d -> b
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[2]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)

        # trg_flow_da
        # flow: d -> a 
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[3]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow_da = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)
        # print('flow_int16_x10_name_trg', flow_int16_x10_name_trg)

        # flow: b -> a 
        file_name = file_name.replace(str(frames[0]).zfill(6), str(frames[2]).zfill(6))
        flow_int16_x10_name_trg = file_name.replace('leftImg8bit.png', str(frames[3]).zfill(6) + '_int16_x10')
        flow_int16_x10_trg = np.load(os.path.join(cfg.TRAIN.flow_path, flow_int16_x10_name_trg + '.npy'))
        trg_flow_b = torch.from_numpy(flow_int16_x10_trg / 10.0).permute(2, 0, 1).unsqueeze(0)

        ##  augmentation  ##
        # flip {b, a}
        flip = random.random() < 0.5
        if flip:
            trg_img_b_wk = torch.flip(trg_img_b, [3])
            trg_img_a_wk = torch.flip(trg_img_a, [3])
            trg_flow_b_wk = torch.flip(trg_flow_b, [3])
        else:
            trg_img_b_wk = trg_img_b
            trg_img_a_wk = trg_img_a
            trg_flow_b_wk = trg_flow_b
        # concatenate {d, c}
        trg_img_concat = torch.cat((trg_img_d, trg_img_c), 2)
        # strong augment {d, c}
        aug = T.Compose([
            T.ToPILImage(),
            T.RandomApply([GaussianBlur(radius=random.choice([5, 7, 9]))], p=0.6),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor()
        ])
        trg_img_concat_st = aug(torch.squeeze(trg_img_concat)).unsqueeze(dim=0)
        # seperate {d, c}
        trg_img_d_st = trg_img_concat_st[:, :, 0:512, :]
        trg_img_c_st = trg_img_concat_st[:, :, 512:, :]

        ## soft_dsp_paste
        if cfg.SOURCE == 'Viper':
            cls_to_use = random.sample(class_to_select, 2)

        elif cfg.SOURCE == 'SynthiaSeq':
            cls_to_use = random.sample(class_to_select, 2)

        # cls_to_use = random.sample(class_to_select, 3)
        # source_image size > target_image_size
        (h, w) = (1280, 640)
        (ch, cw) = (1024, 512)
        

        MixMask = torch.zeros_like(src_label)
        classes = torch.unique(src_label_last)
        nclasses = classes.shape[0]
        classes_ = (classes[torch.Tensor(np.random.choice(nclasses, 2 ,replace=False)).long()]) # int((nclasses+nclasses%2)/2)

        classes = torch.unique(torch.cat((classes_, torch.tensor(class_to_select).float()),0))

        MixMask = generate_class_mask(src_label_last, classes).unsqueeze(0) # .cuda(device)
        MixMask_lam = MixMask.clone() * lam
        
        MixMask_kf = torch.zeros_like(src_label)
        # print('src_label_last_kf', src_label_last_kf)
        MixMask_kf = generate_class_mask(src_label_last_kf, classes).unsqueeze(0) # .cuda(device)
        MixMask_lam_kf = MixMask_kf.clone() * lam


        if cfg.SOURCE == 'Viper':
            sam_14 = random.random() > 0.3
            if sam_14 ==1 or 14 in classes:
                x1 = random.randint(100, w - cw)
                y1 = random.randint(0, h - ch)
            else:
                x1 = random.randint(0, w - cw)
                y1 = random.randint(0, h - ch)
        else:
            sam_14 = None
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)

        # print('x1', x1)
        # print('y1', y1)

        # rescale {d, c}
        # scale_ratio = np.random.randint(100.0 * cfg.TRAIN.SCALING_RATIO[0], 100.0 * cfg.TRAIN.SCALING_RATIO[1]) / 100.0
        # trg_scaled_size = (round(input_size_target[1] * scale_ratio / 8) * 8, round(input_size_target[0] * scale_ratio / 8) * 8)
        # trg_interp_sc = nn.Upsample(size=trg_scaled_size, mode='bilinear', align_corners=True)
        # trg_img_d_st = trg_interp_sc(trg_img_d_st)
        # trg_img_c_st = trg_interp_sc(trg_img_c_st)
        ##  Temporal Pseudo Supervision  ##
        # Cross Frame Pseudo Label
        with torch.no_grad():
            if i_iter < 8000:
                trg_pred_aux, trg_pred, _, _, _, _ = model_tps(trg_img_b_wk.cuda(device), kf = trg_img_a_wk.cuda(device), flow = trg_flow_b_wk, device = device)
            else:
                trg_pred_aux, trg_pred, _, _, _, _, _, _ = model(trg_img_b_wk.cuda(device), kf = trg_img_a_wk.cuda(device), flow = trg_flow_b_wk, device = device)
            # softmax
            trg_prob = F.softmax(trg_pred, dim=1)
            trg_prob_aux = F.softmax(trg_pred_aux, dim=1)
            ### warp
            ### if flip:
            ###     trg_flow = torch.flip(trg_flow, [3])

            interp_flow = nn.Upsample(size=(trg_prob.shape[-2], trg_prob.shape[-1]), mode='bilinear', align_corners=True)
            interp_flow_ratio = trg_prob.shape[-2] / trg_flow.shape[-2]
            trg_flow_warp = (interp_flow(trg_flow) * interp_flow_ratio).float().cuda(device)
            trg_prob_warp = warp_bilinear(trg_prob, trg_flow_warp)
            trg_prob_warp_aux = warp_bilinear(trg_prob_aux, trg_flow_warp)
            # pseudo label
            trg_pl_65 = torch.argmax(trg_prob_warp, 1)
            trg_pl_aux_65 = torch.argmax(trg_prob_warp_aux, 1)
            if flip:
                trg_pl_65 = torch.flip(trg_pl_65, [2])
                trg_pl_aux_65 = torch.flip(trg_pl_aux_65, [2])

            trg_pred_512 = interp_target(trg_pred)
            trg_pred_aux_512 = interp_target(trg_pred_aux)

            trg_prob_512 = F.softmax(trg_pred_512, dim=1)
            trg_prob_aux_512 = F.softmax(trg_pred_aux_512, dim=1)
            interp_flow_512 = nn.Upsample(size=(trg_prob_512.shape[-2], trg_prob_512.shape[-1]), mode='bilinear', align_corners=True)
            interp_flow_ratio_512 = trg_prob_512.shape[-2] / trg_flow.shape[-2]
            trg_flow_warp_512 = (interp_flow_512(trg_flow) * interp_flow_ratio_512).float().cuda(device)
            trg_prob_warp = warp_bilinear(trg_prob_512, trg_flow_warp_512)
            trg_prob_warp_aux = warp_bilinear(trg_prob_aux_512, trg_flow_warp_512)

            
            trg_pl_aux = torch.argmax(trg_prob_warp_aux, 1)
            # trg_pl = torch.argmax(trg_prob_warp, 1)
            # trg_pl_kf = torch.argmax(trg_pred_512, 1) # !!!!! if yita ==1, image c actually is image b


            trg_pl_logits, trg_pl = torch.max(trg_prob_warp, dim=1)
            trg_pl_kf_logits, trg_pl_kf = torch.max(trg_pred_512, dim=1)

            if flip:
                trg_pl = torch.flip(trg_pl, [2])
                trg_pl_aux = torch.flip(trg_pl_aux, [2])
                trg_pl_kf = torch.flip(trg_pl_kf, [2])
                
                trg_pl_logits = torch.flip(trg_pl_logits, [2])
                trg_pl_kf_logits = torch.flip(trg_pl_kf_logits, [2])

            # constructing high confidence target clas-aware template different from source2 class

            classes_t = torch.unique(trg_pl).float()
            # print('classes_t1', classes_t) # [ 0.,  1.,  2.,  7.,  8., 11.]
            # print('classes_', classes_) # [11.,  3.]

            classes_ = classes_.cuda(device)
            b = torch.nonzero((classes_t != classes_[0]) * (classes_t != classes_[1] ), as_tuple=False).squeeze()
            # print('b', b) # [0, 1, 2, 3, 4]

            classes_t = torch.index_select(classes_t, dim=0, index=b)
            # print('classes_t2', classes_t) # [0., 1., 2., 7., 8.]

            nclasses_t = classes_t.shape[0]

            try :
                classes_t = (classes_t[torch.Tensor(np.random.choice(nclasses_t, 2 ,replace=False)).long()])
            except:
                classes_t = (classes_t[torch.Tensor(np.random.choice(nclasses_t, 1 ,replace=False)).long()])

            # print('classes_t3', classes_t) # [8., 1.]

            mask_d = torch.zeros_like(trg_pl)
            mask_d[trg_pl_logits>0.9]=1
            mask_d = generate_class_mask(trg_pl.float(), classes_t) * mask_d

            mask_c = torch.zeros_like(trg_pl_kf)
            mask_c[trg_pl_kf_logits>0.9]=1
            mask_c = generate_class_mask(trg_pl_kf.float(), classes_t) * mask_c


            # # rescale param
            trg_interp_sc2ori = nn.Upsample(size=(trg_pred.shape[-2], trg_pred.shape[-1]), mode='bilinear', align_corners=True)
            
            # trg_pl_kf , _ = model(trg_img_c.cuda(device), pre_f = False)
            # trg_pl_kf = interp_target(trg_pl_kf)
            # trg_pl_kf = F.softmax(trg_pl_kf, dim=1)
            # trg_pl_kf = torch.argmax(trg_pl_kf, 1)

        ## 先硬粘贴 dhp
        # src_cf_last = src_cf_last[:, :, x1:x1+cw, y1:y1+ch]
        # src_label_last = src_label_last[:, x1:x1+cw, y1:y1+ch]
        # MixMask_lam = MixMask_lam[:, x1:x1+cw, y1:y1+ch]
        # MixMask = MixMask[:, x1:x1+cw, y1:y1+ch]
        # src_label = src_label[:, x1:x1+cw, y1:y1+ch]

        # print('src_cf_last[:, :, x1:x1+cw, y1:y1+ch].shape', src_cf_last[:, :, x1:x1+cw, y1:y1+ch].shape) # [1, 3, 512, 1024]
        # print('src_label_last[:, x1:x1+cw, y1:y1+ch].shape', src_label_last[:, x1:x1+cw, y1:y1+ch].shape) # [1, 512, 1024]
        # print('MixMask_lam[:, x1:x1+cw, y1:y1+ch].shape', MixMask_lam[:, x1:x1+cw, y1:y1+ch].shape) # [1, 512, 1024]
        # print('MixMask[:, x1:x1+cw, y1:y1+ch].shape', MixMask[:, x1:x1+cw, y1:y1+ch].shape) # [1, 512, 1024]
        # print('src_label[:, x1:x1+cw, y1:y1+ch].shape', src_label[:, x1:x1+cw, y1:y1+ch].shape) # [1, 512, 1024]

        
        inputs_s_t_d, targets_s_t_d, path_list_d, Masks_longtail = Class_mix(cfg, src_cf_last[:, :, x1:x1+cw, y1:y1+ch], trg_img_d_st, src_label_last[:, x1:x1+cw, y1:y1+ch],
                                                            trg_pl.cpu().float(), MixMask_lam[:, x1:x1+cw, y1:y1+ch], MixMask[:, x1:x1+cw, y1:y1+ch], gta5_cls_mixer, cls_to_use, x1, y1, ch, cw, patch_re=True, sam_14 = sam_14)
        # print('111111111')
        inputs_s_t_c, targets_s_t_c = Class_mix(cfg, src_kf_last[:, :, x1:x1+cw, y1:y1+ch], trg_img_c_st, src_label_last_kf[:, x1:x1+cw, y1:y1+ch],
                                                            trg_pl_kf.cpu().float(), MixMask_lam_kf[:, x1:x1+cw, y1:y1+ch], MixMask_kf[:, x1:x1+cw, y1:y1+ch], gta5_cls_mixer, cls_to_use, x1, y1, ch, cw, patch_re=False, path_list=path_list_d)
        # print('222222222')
        inputs_s_s_cf, targets_s_s_cf = Class_mix(cfg, src_cf_last[:, :, x1:x1+cw, y1:y1+ch], src_img_cf[:, :, x1:x1+cw, y1:y1+ch], src_label_last[:, x1:x1+cw, y1:y1+ch],
                                                            src_label[:, x1:x1+cw, y1:y1+ch], MixMask_lam[:, x1:x1+cw, y1:y1+ch], MixMask[:, x1:x1+cw, y1:y1+ch], gta5_cls_mixer, cls_to_use, x1, y1, ch, cw, patch_re=False, path_list=path_list_d)
        # print('333333333')
        inputs_s_s_kf, targets_s_s_kf = Class_mix(cfg, src_kf_last[:, :, x1:x1+cw, y1:y1+ch], src_img_kf[:, :, x1:x1+cw, y1:y1+ch], src_label_last_kf[:, x1:x1+cw, y1:y1+ch],
                                                            src_label_kf[:, x1:x1+cw, y1:y1+ch], MixMask_lam_kf[:, x1:x1+cw, y1:y1+ch], MixMask_kf[:, x1:x1+cw, y1:y1+ch], gta5_cls_mixer, cls_to_use, x1, y1, ch, cw, patch_re=False, path_list=path_list_d)
        

        mixed_flow_st = Class_mix_flow(cfg, MixMask_kf[:, x1:x1+cw, y1:y1+ch], src_flow_last_cd[:, :, x1:x1+cw, y1:y1+ch], trg_flow_d)
        mixed_flow_ss = Class_mix_flow(cfg, MixMask_kf[:, x1:x1+cw, y1:y1+ch], src_flow_last_cd[:, :, x1:x1+cw, y1:y1+ch], src_flow[:, :, x1:x1+cw, y1:y1+ch])

        
        inputs_s_t_d, targets_s_t_d = Class_mix_nolongtail(cfg, tar_d_last, inputs_s_t_d.cpu(), trg_pl_last.cpu().float(), targets_s_t_d.cpu(), mask_d_last.cpu().float())
        # print('111111111')
        inputs_s_t_c, targets_s_t_c = Class_mix_nolongtail(cfg, tar_c_last, inputs_s_t_c.cpu(), trg_pl_kf_last.cpu().float(), targets_s_t_c.cpu(), mask_c_last.cpu().float())
        # print('222222222')
        inputs_s_s_cf, targets_s_s_cf = Class_mix_nolongtail(cfg, tar_d_last, inputs_s_s_cf.cpu(), trg_pl_last.cpu().float(), targets_s_s_cf.cpu(), mask_d_last.cpu().float())
        # print('333333333')
        inputs_s_s_kf, targets_s_s_kf = Class_mix_nolongtail(cfg, tar_c_last, inputs_s_s_kf.cpu(), trg_pl_kf_last.cpu().float(), targets_s_s_kf.cpu(), mask_c_last.cpu().float())
        

        # print('mask_c_last', mask_c_last)
        # print('tar_dc_flow_last', tar_dc_flow_last)
        # print('mixed_flow_st', mixed_flow_st)
        mixed_flow_st = Class_mix_flow(cfg, mask_c_last.cpu().float(), tar_dc_flow_last.cpu(), mixed_flow_st)
        mixed_flow_ss = Class_mix_flow(cfg, mask_c_last.cpu().float(), tar_dc_flow_last.cpu(), mixed_flow_ss)

        # print('Masks_longtail', Masks_longtail)
        # print('mask_d', mask_d)
        Masks_fused = MixMask[:, x1:x1+cw, y1:y1+ch].float().cuda(device) + Masks_longtail + mask_d.float()
        # print('Masks_fused', Masks_fused)
        Masks_fused[Masks_fused>=1]=1
        MixMask_ = Masks_fused
        # print('MixMask_', MixMask_)
        # print('inputs_s_s_cf.shape', inputs_s_s_cf.shape) # [1, 3, 512, 1024]
        # print('inputs_s_s_kf.shape', inputs_s_s_kf.shape) # [1, 3, 512, 1024]
        # print('MixMask_.shape', MixMask_.shape) # [1, 512, 1024]
        _, src_pred, _, _, _, _, cf_feat, cf_layer4_feat = model(inputs_s_s_cf.cuda(device), kf = inputs_s_s_kf.cuda(device), flow = mixed_flow_ss, device = device, Mask= MixMask_, Masks_lt= Masks_longtail, interp_target = interp_target, fusio = True)
        # # if fusio = True， 前7项的size是512,1024；否则65,129
        # print('src_pred.shape', src_pred.shape) # [1, 12, 512, 1024]
        # print('src_label[:, x1:x1+cw, y1:y1+ch].shape', src_label[:, x1:x1+cw, y1:y1+ch].shape) # [1, 512, 1024]
        # print('targets_s_s_cf.shape', targets_s_s_cf.shape) # [1, 512, 1024]

        loss_src_p = loss_calc(src_pred, src_label[:, x1:x1+cw, y1:y1+ch], device) * (1-lam) + lam * loss_calc(src_pred, targets_s_s_cf, device)

        # print('inputs_s_t_d.shape', inputs_s_t_d.shape) # [1, 3, 512, 1024]
        # print('inputs_s_t_c.shape', inputs_s_t_c.shape) # [1, 3, 512, 1024]
        # print('MixMask[:, x1:x1+cw, y1:y1+ch].shape', MixMask[:, x1:x1+cw, y1:y1+ch].shape) # [1, 512, 1024]

        _, src_pred, _, _, _, _, _, _ = model(inputs_s_t_d.cuda(device), kf = inputs_s_t_c.cuda(device), flow = mixed_flow_st, device = device, mix_layer4_feat = cf_layer4_feat, i_iters = i_iter, Mask= MixMask_, Masks_lt= Masks_longtail, interp_target = interp_target, fusio = True)
        # # if fusio = True， 前7项的size是512,1024；否则65,129
        # print('src_pred.shape', src_pred.shape) # [1, 12, 512, 1024]
        # print('trg_pl.shape', trg_pl.shape) # [1, 512, 1024]
        # print('targets_s_t_d.shape', targets_s_t_d.shape) # [1, 512, 1024]
        
        loss_tar_p = loss_calc(src_pred, trg_pl, device) * (1-lam) + lam * loss_calc(src_pred, targets_s_t_d, device)

        # print('trg_img_d_st.shape', trg_img_d_st.shape) # [1, 3, 512, 1024]
        # print('trg_img_c_st.shape', trg_img_c_st.shape) # [1, 3, 512, 1024]
        # forward prop
        trg_pred_aux, trg_pred, _, _, _, _, _, _ = model(trg_img_d_st.cuda(device), kf = trg_img_c_st.cuda(device), flow = trg_flow_d, device = device)
        # rescale
        trg_pred = trg_interp_sc2ori(trg_pred)
        trg_pred_aux = trg_interp_sc2ori(trg_pred_aux)
        # unsupervised loss
        # print('trg_pred.shape', trg_pred.shape) # [1, 12, 65, 129]
        # print('trg_pl_65.shape', trg_pl_65.shape) # [1, 65, 129]

        objective_vectors_tar_dc = torch.zeros([cfg.NUM_CLASSES, cfg.NUM_CLASSES]).cuda(device)
        vectors, ids = class_features.calculate_mean_vector(trg_pred, trg_pred)
        for t in range(len(ids)):
            objective_vectors_tar_dc = update_objective_SingleVector(objective_vectors_tar_dc, ids[t], vectors[t])
            # if t ==0:
            #     print('objective_vectors_tar_dc[t]', objective_vectors_tar_dc[t])

        loss_trg = loss_calc(trg_pred, trg_pl_65, device)
        if cfg.TRAIN.MULTI_LEVEL:
            # print('trg_pred_aux.shape', trg_pred_aux.shape) # [1, 12, 65, 129]
            # print('trg_pl_aux_65.shape', trg_pl_aux_65.shape) # [1, 65, 129]
            loss_trg_aux = loss_calc(trg_pred_aux, trg_pl_aux_65, device)
        else:
            loss_trg_aux = 0

        _, trg_pred1, _, _, _, _, _, _ = model(trg_img_d.cuda(device), trg_img_a.cuda(device), trg_flow_da, device)
        trg_pred1 = trg_interp_sc2ori(trg_pred1)
        objective_vectors_tar_da = torch.zeros([cfg.NUM_CLASSES, cfg.NUM_CLASSES]).cuda(device)

        vectors, ids = class_features.calculate_mean_vector(trg_pred1, trg_pred1)
        for t in range(len(ids)):
            objective_vectors_tar_da = update_objective_SingleVector(objective_vectors_tar_da, ids[t], vectors[t])
            # if t ==0:
            #     print('objective_vectors_tar_da[t]', objective_vectors_tar_da[t])
        target_temporal = temporal_moudle(objective_vectors_tar_dc, trg_pred, objective_vectors_tar_da, trg_pred1)

        loss_mmd = mmd_rbf(objective_vectors_sou, target_temporal)

        loss = cfg.TRAIN.LAMBDA_T * (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_trg + cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_aux + loss_tar_p) + loss_src_p + loss_sou + 0.01 * loss_mmd
        loss.backward()

        ####  step  ####
        optimizer.step()
        
        ####  logging  ####
        if cfg.TRAIN.MULTI_LEVEL:
            current_losses = {'loss_src': cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main,
                              'loss_src_aux': cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux,
                              'loss_src_p': loss_src_p,
                              'loss_trg': cfg.TRAIN.LAMBDA_T * cfg.TRAIN.LAMBDA_SEG_MAIN * loss_trg,
                              'loss_trg_aux': cfg.TRAIN.LAMBDA_T * cfg.TRAIN.LAMBDA_SEG_AUX * loss_trg_aux,
                              'loss_tar_p': cfg.TRAIN.LAMBDA_T * loss_tar_p,
                              'loss_mmd': 0.01 * loss_mmd
                             }
        else:
            current_losses = {'loss_src': cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main,
                              'loss_src_p': loss_src_p,
                              'loss_trg': cfg.TRAIN.LAMBDA_T * cfg.TRAIN.LAMBDA_SEG_MAIN * loss_trg,
                              'loss_tar_p': cfg.TRAIN.LAMBDA_T * loss_tar_p,
                              'loss_mmd': 0.01 * loss_mmd
                             }
        print_losses(current_losses, i_iter)
        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            print('exp =', cfg.TRAIN.SNAPSHOT_DIR_DSP_CD)
            snapshot_dir = Path(cfg.TRAIN.SNAPSHOT_DIR_DSP_CD)
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')
            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        if viz_tensorboard:
            log_losses_tensorboard(writer, current_losses, i_iter)


        src_cf_last = src_cf.clone()
        src_kf_last = src_kf.clone()
        src_label_last = src_label.clone()
        src_label_last_kf = src_label_kf.clone()
        src_flow_last_cd = src_flow.clone()

        mask_c_last = mask_c
        mask_d_last = mask_d
        tar_dc_flow_last = trg_flow_warp_512
        tar_d_last = trg_img_d.clone()
        tar_c_last = trg_img_c.clone()
        trg_pl_last = trg_pl
        trg_pl_kf_last = trg_pl_kf
## utils
def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')

def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)

def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()

def hist_match(img_src, img_trg):
    import skimage
    from skimage import exposure
    img_src = np.asarray(img_src.squeeze(0).transpose(0, 1).transpose(1, 2), np.float32)
    img_trg = np.asarray(img_trg.squeeze(0).transpose(0, 1).transpose(1, 2), np.float32)
    images_aug = exposure.match_histograms(img_src, img_trg, multichannel=True)
    return torch.from_numpy(images_aug).transpose(1, 2).transpose(0, 1).unsqueeze(0)
    
class GaussianBlur(object):

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.radius))
        

class EMA(object):

    def __init__(self, model, alpha=0.999):
        """ Model exponential moving average. """
        self.step = 0
        self.model = model
        self.alpha = alpha
        self.shadow = self.get_model_state()
        self.backup = {}
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        # NOTE: Buffer values are for things that are not parameters,
        # such as batch norm statistics
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]

    def update_params(self):
        decay = self.alpha
        state = self.model.state_dict()  # current params
        for name in self.param_keys:
            self.shadow[name].copy_(
                    decay * self.shadow[name] + (1 - decay) * state[name])
        self.step += 1

    def update_buffer(self):
        # No EMA for buffer values (for now)
        state = self.model.state_dict()
        for name in self.buffer_keys:
            self.shadow[name].copy_(state[name])

    def apply_shadow(self):
        self.backup = self.get_model_state()
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.backup)

    def get_model_state(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }

class Class_Features:
    def __init__(self, device, numbers = 15):
        self.class_numbers = numbers
        self.device = device

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(self.device)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(self.device))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def calculate_mean_vector(self, feat_cls, outputs, labels=None, thresh=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        if thresh is None:
            thresh = -1
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = conf.ge(thresh).float()
        # print('torch.sum(mask)', torch.sum(mask))
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        # print('outputs_argmax.shape', outputs_argmax.shape)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels is None:
            outputs_pred = outputs_argmax #[1,12,65,129]
        else:
            labels_expanded = self.process_label(labels)
            outputs_pred = labels_expanded * outputs_argmax

        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1) #[1,12,1,1]
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 5:
                    continue
                s = feat_cls[n] * outputs_pred[n][t] * mask[n] # [256,65,129]*[65,129]=[256,65,129]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = torch.sum(torch.sum(s, dim=-1), dim=-1) / torch.sum(outputs_pred[n][t] * mask[n])
                # s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t] # [256,1,1]/[1,1]=[256,1,1]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids


def update_objective_SingleVector(objective_vectors, id, vector):
    if vector.sum().item() == 0:
        return objective_vectors

    objective_vectors[id] = vector.squeeze()
    
    return objective_vectors


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss



def temporal_moudle(objective_vectors_tar_dc, trg_pred, objective_vectors_tar_da, trg_pred1):
    tem_weighted = False # False
    if tem_weighted:
        w_db_sou = torch.mean(prob_2_entropy(F.softmax(trg_pred, dim=1)))
        w_da_sou = torch.mean(prob_2_entropy(F.softmax(trg_pred1, dim=1)))
        wei_db_sou = torch.exp(w_db_sou)/(torch.exp(w_db_sou)+torch.exp(w_da_sou))
        wei_da_sou = torch.exp(w_da_sou)/(torch.exp(w_db_sou)+torch.exp(w_da_sou))
    else:
        wei_db_sou, wei_da_sou = 0.5, 0.5
    # print('wei_db_sou', wei_db_sou)
    # print('wei_da_sou', wei_da_sou)

    target_temporal = wei_db_sou * objective_vectors_tar_dc + wei_da_sou * objective_vectors_tar_da

    return target_temporal