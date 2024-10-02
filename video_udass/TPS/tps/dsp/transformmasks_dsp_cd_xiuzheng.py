import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import json
from tps.utils import project_root
import imageio
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg') 
import random

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output+1e-8, target.long())
        loss = torch.mean(loss * pixelWiseWeight)
        return loss


def oneMix(cfg, mask, data = None, target = None):
    #Mix
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

    # print(mask.device)
    # print(mask.float().device)

    if not (data is None):
        stackedMask0 = torch.broadcast_tensors(mask.float(), data[0])[0]
        # print('stackedMask0.shape', stackedMask0.shape) # [3, 512, 1024]
        # print('mask.shape', mask.shape) # [1, 512, 1024]
        # print('data.shape', data.shape) # [2, 3, 512, 1024]
        # print('stackedMask0image.sum()', stackedMask0.sum()) # [3, 512, 1024]

        # plt.figure()
        # plt.imshow((255*stackedMask0).cpu().numpy().transpose(1,2,0).astype(np.uint8))
        # plt.savefig('./stackedMask0_image_'+str(2)+'.png')
        # plt.close()

        # a=stackedMask0*data[0]
        # b=(1-stackedMask0)*data[1]
        # print(stackedMask0.device) # cpu
        # print(data.device) # cpu
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)

        # plt.figure()
        # plt.imshow((a.cpu().numpy().transpose(1,2,0)).astype(np.uint8))
        # plt.savefig('./images0*m_'+str(2)+'.png')
        # plt.close()

        # plt.figure()
        # plt.imshow((b.cpu().numpy().transpose(1,2,0)).astype(np.uint8))
        # plt.savefig('./images1*m_'+str(2)+'.png')
        # plt.close()

    if not (target is None):
        stackedMask0 = torch.broadcast_tensors(mask.float(), target[0])[0]
        # m = stackedMask0.unsqueeze(0)
        # # print('stackedMask0label.sum()', stackedMask0.sum()) # [512, 1024]
        # plt.figure()
        # plt.imshow(255*torch.cat([m,m,m],dim=0).cpu().numpy().transpose(1,2,0).astype(np.uint8))
        # plt.savefig('./stackedMask0_label_'+str(2)+'.png')
        # plt.close()

        # b = stackedMask0*target[0].squeeze()
        # c = (1-stackedMask0)*target[1].squeeze()

        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1])

        # amax_output_col = colorize_mask(np.asarray(b.cpu(), dtype=np.uint8))
        # amax_output_col.save('./targets0*m_'+str(2)+'.png')

        
        # amax_output_col = colorize_mask(np.asarray(c.cpu(), dtype=np.uint8))
        # amax_output_col.save('./targets1*m_'+str(2)+'.png')

    return data, target


def generate_class_mask(pred, classes):  #  [h, w]  [n]
    # print('pred.shape', pred.shape)
    # print('classes', classes)
    # print('classes.shape', classes.shape)
    pred, classes = torch.broadcast_tensors(pred.squeeze().unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))  # [1, h, w]  [n, 1, 1] -- > pred: [n, h, w]  classes: [n, h, w]
    # print('pred.shape', pred.shape)
    # print('classes.shape', classes.shape)
    # print('pred', pred)
    # print('classes', classes)
    N = pred.eq(classes.float()).sum(0)
    # print('N.shape', N.shape)
    return N # 返回指定类别classes的mask


def Class_mix(cfg, image1, image2, label1, label2, mask_img, mask_lbl, cls_mixer, cls_list, x1=None, y1=None, ch=None, cw=None, patch_re=True, path_list=None, sam_14 = None):
    #                         源域，  目标域，源域标签，目标域标签，
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

    # print('image1.shape', image1.shape) # [1, 3, 512, 1024]
    # print('image2.shape', image2.shape) # [1, 3, 512, 1024]
    # print('label1.shape', label1.shape) # [1, 512, 1024]
    # print('label2.shape', label2.shape) # [1, 512, 1024]
    # print('mask_img.shape', mask_img.shape) # [1, 512, 1024]
    # print('mask_lbl.shape', mask_lbl.shape) # [1, 512, 1024]
    # print('torch.cat((image1, image2)).shape', torch.cat((image1, image2)).shape)  # [2, 3, 512, 1024]
    # print('torch.cat((label1, label2)).shape', torch.cat((label1, label2)).shape)  # [2, 512, 1024]

    # print('cls_to_use2', cls_list) # [3, 5]
    inputs_, _ = oneMix(cfg, mask_img, data=torch.cat((image1, image2)))
    _, targets_ = oneMix(cfg, mask_lbl, target=torch.cat((label1, label2)))
    # print('inputs_.shape', inputs_.shape) # [1, 3, 512, 1024]
    # print('targets_.shape', targets_.shape) # [1, 512, 1024]
    # a = inputs_[0].squeeze().permute(1, 2, 0)
    # plt.figure()
    # plt.imshow((a.cpu().numpy()).astype(np.uint8))
    # plt.savefig('./inputs0_'+ str(2)+ '.png')
    # plt.close()

    # b = targets_[0].squeeze()
    # amax_output_col = colorize_mask(np.asarray(b, dtype=np.uint8))
    # amax_output_col.save('./targets0_'+str(2)+'.png')
    if patch_re == True:
        inputs_, targets_, path_list, Masks_longtail = cls_mixer.mix(cfg, inputs_.squeeze(0), cls_list, x1, y1, ch, cw, patch_re, in_lbl=targets_, path_list=None, sam_14 = None)
        return inputs_, targets_.unsqueeze(0), path_list, Masks_longtail
    else:
        inputs_, targets_ = cls_mixer.mix(cfg, inputs_.squeeze(0), cls_list, x1, y1, ch, cw, patch_re, in_lbl=targets_, path_list=path_list, sam_14 = None)
        return inputs_, targets_.unsqueeze(0)
    # out_img, out_lbl = strongTransform_ammend(strong_parameters, data=inputs_, target=targets_)
    # return out_img, out_lbl
    # print('inputs_.shape', inputs_.shape)
    # print('targets_.shape', targets_.shape)
    


def Class_mix_nolongtail(cfg, image1, image2, label1, label2, mask_lbl):

    inputs_, _ = oneMix(cfg, mask_lbl, data=torch.cat((image1, image2)))
    _, targets_ = oneMix(cfg, mask_lbl, target=torch.cat((label1, label2)))
    
    return inputs_, targets_


def Class_mix_flow(cfg, mask_flow=None, src_flow_last_cd=None, trg_flow=None):
    # print('src_flow_last_cd.shape', src_flow_last_cd.shape)
    # print('trg_flow.shape', trg_flow.shape)
    # print('mask_flow.shape', mask_flow.shape)


    mixed_flow, _ = oneMix(cfg, mask_flow, data = torch.cat((src_flow_last_cd.float(),trg_flow.float())))
    # print('mixed_flow.shape', mixed_flow.shape)
    return mixed_flow


class rand_mixer():
    def __init__(self, dataset, device):
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.device = device
        self.dataset = dataset
        if self.dataset == "viper_seq":
            jpath = str(project_root / 'data/viper_ids2path.json')
            self.class_map = {3: 0, 4: 1, 9: 2, 11: 3, 13: 4, 14: 5, 7: 6, 8: 6, 6: 7, 2: 8, 20: 9, 24: 10, 27: 11,
                          26: 12, 23: 13, 22: 14}
            self.ignore_ego_vehicle = True
            self.root = str(project_root / 'data/Viper')
            self.image_size = (1280, 720)
            self.labels_size = self.image_size
        elif self.dataset == "synthia_seq":
            jpath = str(project_root / 'data/synthia_ids2path.json')
            self.class_map = {3: 0, 4: 1, 2: 2, 5: 3, 7: 4, 15: 5, 9: 6, 6: 7, 1: 8, 10: 9, 11: 10, 8: 11,}
            self.ignore_ego_vehicle = False
            self.root = str(project_root / 'data/Cityscapes')
            self.image_size = (1280, 760)
            self.labels_size = self.image_size
        else:
            print('rand_mixer {} unsupported'.format(self.dataset))
            return
        
        
        
        with open(jpath, 'r') as load_f:
            self.ids2img_dict = json.load(load_f)


    def get_image(self, file):
        return self._load_img(file, self.image_size, Image.BICUBIC, rgb=True)

    def get_labels(self, file):
        return self._load_img(file, self.labels_size, Image.NEAREST, rgb=False)

    def get_labels_synthia_seq(self, file):
        # img = Image.open(file)
        lbl = imageio.imread(file, format='PNG-FI')[:, :, 0]
        img = Image.fromarray(lbl)
        img = img.resize(self.labels_size, Image.NEAREST)
        return np.asarray(img, np.float32)

    def _load_img(self, file, size, interpolation, rgb):
        
        img = Image.open(file)
        if rgb:
            img = img.convert('RGB')
        img = img.resize(size, interpolation)
        return np.asarray(img, np.float32)


    def preprocess(self, image):
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        return image.transpose((2, 0, 1))

    def mix(self, cfg, in_img, classes, x1, y1, ch, cw, patch_re, in_lbl=None, path_list=None, sam_14 = None):
        # print('cls_to_use3', classes)
        a_c = 0

        if patch_re:
            path_list=[]
            # print('----------------classes-----longtail-----------\n', classes)
            if cfg.SOURCE == 'Viper' and sam_14:
                # print('----------------classes-----longtail-----------\n', classes)
                classes.append(14)
        classes = np.unique(classes)
        # print('----------------classeses-----longtail-----------\n', classes)
        if patch_re == True:
            Masks_longtail = torch.zeros_like(in_lbl).cuda(self.device)

        cls_num_ = 0
        for i in classes:
            # print('i111', i)
            cls_num_ += 1
            if patch_re == True:
                while(True):
                    # print('---i---', i)
                    name = random.sample(self.ids2img_dict[str(i)], 1)
                    # if i == 14:
                    #     print('name', name)
                    #print('name', name)
                    if self.dataset == "viper_seq":
                        img_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'train/img' , name[0])

                        image = self.get_image(img_path)
                        
                        label_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'train/cls' , name[0].replace('jpg','png'))
                        label = self.get_labels(label_path)
                        if self.ignore_ego_vehicle:
                            lbl_car = label == 24
                            ret, lbs, stats, centroid = cv2.connectedComponentsWithStats(np.uint8(lbl_car))
                            lb_vg = lbs[-1, lbs.shape[1] // 2]
                            if lb_vg > 0:
                                label[lbs == lb_vg] = 0

                    elif self.dataset == "synthia_seq":
                        img_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'rgb', name[0])
                        image = self.get_image(img_path)[:-120, :, :]

                        label_path = os.path.join(cfg.DATA_DIRECTORY_SOURCE, 'label', name[0])
                        label = self.get_labels_synthia_seq(label_path)[:-120, :]
                        
                    label_copy = 255 * np.ones(label.shape, dtype=np.float32)
                    for k, v in self.class_map.items():
                        label_copy[label == k] = v

                    image = self.preprocess(image)
                    # print('image.shape', image.shape)
                    if i ==14:
                        if torch.sum(generate_class_mask(torch.Tensor(label_copy.copy())[x1:x1+cw, y1:y1+ch], torch.Tensor([i]).type(torch.int64))) > 100 :
                            break
                    else:
                        if torch.sum(generate_class_mask(torch.Tensor(label_copy.copy())[x1:x1+cw, y1:y1+ch], torch.Tensor([i]).type(torch.int64))) > 5 :
                            break

                img = torch.Tensor(image.copy()).unsqueeze(0).cuda(self.device)
                lbl = torch.Tensor(label_copy.copy()).cuda(self.device)
                
                path_list.append(img)
                path_list.append(lbl)
                # print('path_list.shape1', np.array(path_list[0].cpu()).shape)
                # print('path_list', path_list)
            elif patch_re == False:
                # print('path_list.shape2', np.array(path_list[1].cpu()).shape)
                # print('len(path_list)', len(path_list))

                img = path_list[a_c]
                a_c += 1
                lbl = path_list[a_c]
                a_c += 1
                # print('a_c', a_c)

            class_i = torch.Tensor([i]).type(torch.int64).cuda(self.device)
            # print('i222', class_i)
            MixMask = generate_class_mask(lbl[x1:x1+cw, y1:y1+ch], class_i).cuda(self.device)
            # print('MixMask.shape', MixMask.shape) # [512, 1024]
            # print('torch.sum(MixMask)', torch.sum(MixMask))
            if patch_re == True:
                Masks_longtail += MixMask.float()
            # print('img[:,:,x1:x1+cw, y1:y1+ch].shape', img[:,:,x1:x1+cw, y1:y1+ch].shape) # [1, 3, 512, 1024]
            # print('in_img.unsqueeze(0).cuda(self.device).shape', in_img.unsqueeze(0).cuda(self.device).shape) # [1, 3, 512, 1024]
            

            if cls_num_ == 1:
                mixdata = torch.cat((img[:,:,x1:x1+cw, y1:y1+ch], in_img.unsqueeze(0).cuda(self.device)))
            elif cls_num_>1:
                mixdata = torch.cat((img[:,:,x1:x1+cw, y1:y1+ch], data.cuda(self.device)))

            # print('mixdata.shape', mixdata.shape) # [2, 3, 512, 1024]
            if in_lbl is None:
                data, _ = oneMix(cfg, MixMask, data=mixdata)
                # print('data.shape', data.shape)
                # return data
            else:
                # print('in_lbl.shape', in_lbl.shape) # [1, 512, 1024]
                # print('lbl[x1:x1+cw, y1:y1+ch].unsqueeze(0).shape', lbl[x1:x1+cw, y1:y1+ch].unsqueeze(0).shape) # [1, 512, 1024]
                if cls_num_ == 1:
                    mixtarget = torch.cat((lbl[x1:x1+cw, y1:y1+ch].unsqueeze(0), in_lbl.cuda(self.device)), 0)
                    # print('torch.sum(lbl[x1:x1+cw, y1:y1+ch])', torch.sum(lbl[x1:x1+cw, y1:y1+ch]))
                elif cls_num_>1:
                    mixtarget = torch.cat((lbl[x1:x1+cw, y1:y1+ch].unsqueeze(0), target.unsqueeze(0).cuda(self.device)), 0)
                    # print('torch.sum(target0)', torch.sum(target))
                # print('torch.sum(in_lbl)', torch.sum(in_lbl))

                # print('mixdata.shape', mixdata.shape) # [2, 3, 512, 1024]
                # print('mixtarget.shape', mixtarget.shape) # [2, 512, 1024]
                data, target = oneMix(cfg, MixMask.float(), data=mixdata, target=mixtarget)

                # print('torch.sum(lbl[x1:x1+cw, y1:y1+ch])', torch.sum(lbl[x1:x1+cw, y1:y1+ch])) #  
                # print('torch.sum(data)', torch.sum(data)) #  
                # print('torch.sum(target1)', torch.sum(target)) #  
                # print('target.unsqueeze(0).shape', target.unsqueeze(0).shape) # [1, 512, 1024]
                # print('data.shape', data.shape) #  

                # print('data.shape', data.shape) # [1, 3, 512, 1024]
                # print('target.shape', target.shape) # [512, 1024]
                # print('in_lbl.shape', in_lbl.shape) # [1, 512, 1024]
                # print('in_img.shape', in_img.shape) # [1, 512, 1024]
                # if patch_re:
                #     #d
                #     # return data, target
                #     # cd
                #     return data, target, path_list
                # else:
                #     return data, target
        

        if in_lbl is None:
            return data
        else:
            if patch_re:
                return data, target, path_list, Masks_longtail
            else:
                return data, target