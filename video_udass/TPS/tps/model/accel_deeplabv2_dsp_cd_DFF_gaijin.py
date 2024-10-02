import torch
import torch.nn as nn
from tps.utils.resample2d_package.resample2d import Resample2d
import torch.nn.functional as F
affine_par = True

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class ClassifierModule_proda(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule_proda, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        self.head = nn.Sequential(*[nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x, inter_s_cf_feat = None, Mask= None, pre_f = True):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)

        out_feat = self.head[0](out)
        # if pre_f == True and inter_s_cf_feat is not None:  #####!!!!!!!! pre_f == False in 另一个accel文件 
        #     out_feat = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)(out_feat)
        #     out_feat = (inter_s_cf_feat * Mask + out_feat * Mask)*0.5 + out_feat * (1-Mask)
        out =self.head[1](out_feat)
        return out, out_feat
        # 如果执行if,返回的out: b,class,512,1024; out_feat: b,256,512,1024
        # 如果不执行if,返回的out: b,class,65,129; out_feat: b,256,65,129

class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        # self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer7 = ClassifierModule_proda(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.sf_layer = self.get_score_fusion_layer(num_classes)
        self.warp_bilinear = Resample2d(bilinear=True) # Propagation

    def get_score_fusion_layer(self, num_classes):
        sf_layer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.zeros_(sf_layer.weight)
        nn.init.eye_(sf_layer.weight[:, :num_classes, :, :].squeeze(-1).squeeze(-1))
        return sf_layer

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, cf, kf = None, flow = None, device = None, mix_layer4_feat = None, i_iters = None, Mask= None, Masks_lt= None, interp_target = None, pre_f = True, fusio = None):
        # print('x00.shape', cf.shape)
        cf = self.conv1(cf)
        # print('x01.shape', cf.shape)
        cf = self.bn1(cf)
        # print('x02.shape', cf.shape)
        cf = self.relu(cf)
        # print('x03.shape', cf.shape)
        cf = self.maxpool(cf)
        # print('x1.shape', cf.shape)
        cf = self.layer1(cf)
        # print('x2.shape', cf.shape)
        cf = self.layer2(cf)
        # print('x3.shape', cf.shape)
        cf = self.layer3(cf)
        if self.multi_level:
            cf_aux = self.layer5(cf)
        else:
            cf_aux = None
        # print('x4.shape', cf.shape)
        cf4 = self.layer4(cf)
        # print('x6.shape', cf4.shape)

        if pre_f == True and mix_layer4_feat is not None and (i_iters>8000 or i_iters<13):
            Mask_ = F.interpolate(Mask.unsqueeze(0), scale_factor=0.25).squeeze()
            # print('Mask.shape', Mask.shape) # [1, 512, 1024]

            # mix_layer4_feat = nn.Upsample(size=(128, 256), mode='bilinear',align_corners=True)(mix_layer4_feat).squeeze()
            
            mix_layer4_feat = F.interpolate(mix_layer4_feat, size=(128, 256), mode='bilinear', align_corners=True).squeeze()
            # print('mix_layer4_feat.shape', mix_layer4_feat.shape) # [1, 256, 65, 129]

            # cf1 = nn.Upsample(size=(128, 256), mode='bilinear',align_corners=True)(cf4).squeeze()
            cf1 = F.interpolate(cf4, size=(128, 256), mode='bilinear', align_corners=True).squeeze()
            # print('cf1.shape', cf1.shape) # [1, 256, 65, 129]

            cf1 = (mix_layer4_feat * Mask_ * 0.3 + cf1 * Mask_ * 0.7 + cf1 * (1-Mask_)).unsqueeze(0)
            # print('x4.shape', cf1.shape)
            
            # cf1 = nn.Upsample(size=(512, 1024), mode='bilinear',align_corners=True)(cf1.unsqueeze(0)).squeeze()
            cf1 = F.interpolate(cf1, size=(65, 129), mode='bilinear', align_corners=True)
            
            # print('cf1.shape', cf1.shape) # [1, 2048, 65, 129]
            # print('x6.shape', cf1.shape)
            cf, cf_feat = self.layer7(cf1)
            # cf = interp_target(cf)
            # cf_feat = interp_target(cf_feat)

        else:
            cf, cf_feat = self.layer7(cf4)
            # cf = interp_target(cf)

        # cf, cf_feat = self.layer7(cf4, inter_s_cf_feat, Mask, pre_f)
        # print('cf_feat.shape', cf_feat.shape)
        # print('cf.shape', cf.shape)
        if pre_f == True:
            with torch.no_grad():
                kf = self.conv1(kf)
                kf = self.bn1(kf)
                kf = self.relu(kf)
                kf = self.maxpool(kf)
                kf = self.layer1(kf)
                kf = self.layer2(kf)
                kf = self.layer3(kf)
                if self.multi_level:
                    kf_aux = self.layer5(kf)
                else:
                    kf_aux = None
                kf1 = self.layer4(kf)
                kf, kf_feat = self.layer7(kf1)
            if fusio == True:
                cf = interp_target(cf)
                cf_feat = interp_target(cf_feat)
                cf_aux = interp_target(cf_aux)
                kf = interp_target(kf)
                kf_aux = interp_target(kf_aux)

            interp_flow2cf = nn.Upsample(size=(kf.shape[-2], kf.shape[-1]), mode='bilinear', align_corners=True)
            interp_flow2cf_ratio = kf.shape[-2] / flow.shape[-2]
            # print('cf_aux.shape', cf_aux.shape)
            # print('kf_aux.shape', kf_aux.shape)
            flow_cf = (interp_flow2cf(flow) * interp_flow2cf_ratio).float().cuda(device)
            # print('flow_cf.shape', flow_cf.shape)
            # print('self.warp_bilinear(kf_aux, flow_cf).shape', self.warp_bilinear(kf_aux, flow_cf).shape)
            # print('torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1).shape', torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1).shape)
            if fusio == True:
                pred_aux = self.sf_layer(torch.cat((cf_aux, (self.warp_bilinear(kf_aux, flow_cf)*(1-Masks_lt) + cf_aux*Masks_lt)), dim=1))
                pred = self.sf_layer(torch.cat((cf, self.warp_bilinear(kf, flow_cf)*(1-Masks_lt)+cf*Masks_lt), dim=1))
            else:
                pred_aux = self.sf_layer(torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1))
                pred = self.sf_layer(torch.cat((cf, self.warp_bilinear(kf, flow_cf)), dim=1))

            return pred_aux, pred, cf_aux, cf, kf_aux, kf, cf_feat, cf4   # if fusio = True， 前7项的size是512,1024；否则65,129
        else:
            return cf, cf_feat

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []
        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        # b.append(self.layer6.parameters())
        b.append(self.layer7.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def get_1x_lr_params_sf_layer(self):
        b = []
        b.append(self.sf_layer.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_1x_lr_params_sf_layer(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]


def get_accel_deeplab_v2(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model
