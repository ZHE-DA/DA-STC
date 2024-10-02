# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import sys
sys.path.append('/home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS')

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.models.decode_heads.isa_head import ISALayer
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.aspp_head import ASPPModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.segformer_head import MLP
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPModule

import numpy as np
from tps.utils.resample2d_package.resample2d import Resample2d
import torch.nn.functional as F

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained='pretrained/mit_b5.pth',
                 init_cfg=None,
                 freeze_patch_embed=False):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) \
        #     if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)

    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class mit_b0(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)



class mit_b1(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b2(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b3(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b4(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


class mit_b5(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)







class ASPPWrapper(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 norm_cfg,
                 act_cfg,
                 align_corners,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels,
                                             **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        # print('x_aspp_in.shape', x.shape)
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


def build_layer(in_channels, out_channels, type, **kwargs):
    # print(type)
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(
            in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


class DAFormerHead(BaseDecodeHead):

    def __init__(self, decoder_params, num_classes, **kwargs):
        super(DAFormerHead, self).__init__(
            input_transform='multiple_select', num_classes = num_classes, decoder_params= decoder_params, **kwargs)

        assert not self.align_corners
        # decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels,
                                             embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(
                    in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(
            sum(embed_dims), self.channels, **fusion_cfg)

        self.sf_layer = self.get_score_fusion_layer(self.num_classes)
        self.warp_bilinear = Resample2d(bilinear=True) # Propagation

    def get_score_fusion_layer(self, num_classes):
        sf_layer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.zeros_(sf_layer.weight)
        nn.init.eye_(sf_layer.weight[:, :num_classes, :, :].squeeze(-1).squeeze(-1))
        return sf_layer

    def forward(self, inputs, mix_layer4_feat=None, Mask=None, fusio=None):  # (cf4, mix_layer4_feat, Mask)
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     mmcv.print_log(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous()\
                    .reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)
        # print('_c[0].shape', _c[0].shape) # [1, 256, 128, 256]
        # print('_c[1].shape', _c[1].shape) # [1, 256, 128, 256]
        # print('_c[2].shape', _c[2].shape) # [1, 256, 128, 256]
        # print('_c[3].shape', _c[3].shape) # [1, 256, 128, 256]
        # print('torch.cat(list(_c.values()), dim=1).shape', torch.cat(list(_c.values()), dim=1).shape) # [1, 1024, 128, 256]
        
        # print('mix_layer4_feat', mix_layer4_feat)
        if mix_layer4_feat is not None:
            x = torch.cat(list(_c.values()), dim=1)
            Mask_ = F.interpolate(Mask.unsqueeze(0), scale_factor=0.25).squeeze()
            # print('Mask.shape', Mask.shape) #  
            # print('Mask_.shape', Mask_.shape) #  
            # print('mix_layer4_feat.shape', mix_layer4_feat.shape) #
            # print('x_c.shape0', x_c.shape) #  
            # mix_layer4_feat = F.interpolate(mix_layer4_feat, size=(128, 256), mode='bilinear', align_corners=True).squeeze()
            # x_c = F.interpolate(x_c, size=(128, 256), mode='bilinear', align_corners=True).squeeze()
            x = (mix_layer4_feat * Mask_ * 0.3 + x * Mask_ * 0.7 + x * (1-Mask_))#.unsqueeze(0)
            # x_c = F.interpolate(x_c, size=(65, 129), mode='bilinear', align_corners=True)
            # print('x_c.shape1', x_c.shape) #
            x = self.fuse_layer(x)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        # print('x.shape', x.shape) # [1, 256, 128, 256]
        x = self.cls_seg(x)
        # print('x.shape1', x.shape) # [1, 19, 128, 256]

        # print('fusio', fusio)
        # print('mix_layer4_feat', mix_layer4_feat)
        if fusio == True and mix_layer4_feat is None:
            # print('situation 2')
            return x, torch.cat(list(_c.values()), dim=1)
        else:
            return x





def DAFormer(cf, kf = None, flow = None, device = None, mix_layer4_feat = None, i_iters = None, Mask= None, Masks_lt= None, interp_target = None, pre_f = True, fusio = None):

    model = mit_b5().to(device)
    # model.init_weights()
    norm_cfg=dict(type='BN', requires_grad=True)
    decoder = DAFormerHead(decoder_params=dict(
                    embed_dims=256,
                    embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                    embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                    fusion_cfg=dict(type='aspp',sep=True,dilations=(1, 6, 12, 18),pool=False,act_cfg=dict(type='ReLU'),norm_cfg=norm_cfg))).to(device)
    cf4 = model(cf)

    # print('cf4[0].shape', cf4[0].shape)
    # print('cf4[1].shape', cf4[1].shape)
    # print('cf4[2].shape', cf4[2].shape)
    # print('cf4[3].shape', cf4[3].shape)
    # print('fusio+', fusio)
    # print('mix_layer4_feat+', mix_layer4_feat)
    if pre_f == True and mix_layer4_feat is not None and (i_iters>8000 or i_iters<13) and fusio == True:
        # print('situation 1+')
        cf = decoder(cf4, mix_layer4_feat, Mask, fusio)
    elif fusio == True and mix_layer4_feat is None:
        # print('situation 2+')
        cf, x_c = decoder(cf4, mix_layer4_feat=None, Mask=None, fusio=True)
    else:
        cf = decoder(cf4)
        # print('situation 3+')
    # print('cf.shape', cf.shape)

    if pre_f == True:
        with torch.no_grad():
            kf = model(kf)
            kf = decoder(kf)
            # print('kf.shape', kf.shape)
        if fusio == True:
            cf = interp_target(cf)
            # cf_feat = interp_target(cf_feat)
            # cf_aux = interp_target(cf_aux)
            kf = interp_target(kf)
            # kf_aux = interp_target(kf_aux)

        # kf_cpu = kf.cpu().numpy()
        interp_flow2cf = nn.Upsample(size=(kf.shape[-2], kf.shape[-1]), mode='bilinear', align_corners=True)
        interp_flow2cf_ratio = kf.shape[-2] / flow.shape[-2]
        # print('cf_aux.shape', cf_aux.shape)
        # print('kf_aux.shape', kf_aux.shape)
        flow_cf = (interp_flow2cf(flow) * interp_flow2cf_ratio).cuda(device)
        # print('flow_cf.shape', flow_cf.shape)
        # print('self.warp_bilinear(kf_aux, flow_cf).shape', self.warp_bilinear(kf_aux, flow_cf).shape)
        # print('torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1).shape', torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1).shape)

        # # import numpy as np
        # # kf_aux_rec = np.zeros(cf_aux.shape)
        # kf_rec = np.zeros(cf.shape)
        # rec_positions = np.zeros(cf.shape)
        # for x in range(cf.shape[-1]):
        #     for y in range(cf.shape[-2]):
        #         x_flow = int(round(x - flow_cf[:, 0, y, x][0]))
        #         y_flow = int(round(y - flow_cf[:, 1, y, x][0]))
        #         if x_flow >= 0 and x_flow < flow_cf.shape[-1] and y_flow >= 0 and y_flow < flow_cf.shape[-2]:
        #             # kf_aux_rec[:, :, y_flow, x_flow] = kf_aux_cpu[:, :, y, x]
        #             kf_rec[:, :, y_flow, x_flow] = kf_cpu[:, :, y, x]
        #             rec_positions[:, :, y_flow, x_flow] = 1
        # # kf_aux_rec = torch.from_numpy(kf_aux_rec)
        # kf_rec = torch.from_numpy(kf_rec).float().cuda(device)
        # rec_positions = torch.from_numpy(rec_positions).float().cuda(device)





        if fusio == True:
            # print('cf.shape', cf.shape)
            # print('Masks_lt.shape', Masks_lt.shape)
            # print('rec_positions.shape', rec_positions.shape)
            # print('kf_rec.shape', kf_rec.shape)
            # pred_aux = self.sf_layer(torch.cat((cf_aux, (self.warp_bilinear(kf_aux, flow_cf)*(1-Masks_lt) + cf_aux*Masks_lt)), dim=1))
            # pred = decoder.sf_layer(torch.cat((cf, (rec_positions*kf_rec)*(1-Masks_lt)+cf*Masks_lt), dim=1))
            pred = decoder.sf_layer(torch.cat((cf, decoder.warp_bilinear(kf, flow_cf)*(1-Masks_lt)+cf*Masks_lt), dim=1))
        else:
            # pred_aux = self.sf_layer(torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1))
            # pred = decoder.sf_layer(torch.cat((cf, decoder.warp_bilinear(kf, flow_cf)), dim=1))
            # pred = decoder.sf_layer(torch.cat((cf, (rec_positions*kf_rec)), dim=1))
            pred = decoder.sf_layer(torch.cat((cf, decoder.warp_bilinear(kf, flow_cf)), dim=1))

        
        if fusio == True and mix_layer4_feat is None:
            return pred, x_c
        else:
            return pred
        
    else:
        return cf




class DAFormer_tps(nn.Module):
    def __init__(self, device=None, pre_f=True, num_classes=None):
        super(DAFormer_tps, self).__init__()
        self.model = mit_b5().to(device)
        # self.model.init_weights()
        norm_cfg = dict(type='BN', requires_grad=True)  
        self.device = device
        self.pre_f = pre_f
        self.num_classes = num_classes
        self.decoder = DAFormerHead(
            num_classes = self.num_classes,
            decoder_params=dict(
                embed_dims=256,
                embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                fusion_cfg=dict(
                    type='aspp',
                    sep=True,
                    dilations=(1, 6, 12, 18),
                    pool=False,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=norm_cfg
                )
            )
        ).to(device)
        
    def forward(self, cf, kf=None, flow=None, mix_layer4_feat=None, i_iters=None, Mask=None, Masks_lt=None, interp_target=None, fusio=None):
        cf = self.model(cf)
        # print(self.num_classes)
        # print('cf[0].shape', cf[0].shape)
        # print('cf[1].shape', cf[1].shape)
        # print('cf[2].shape', cf[2].shape)
        # print('cf[3].shape', cf[3].shape)
        # print('fusio+', fusio)
        # print('mix_layer4_feat+', mix_layer4_feat)
        if self.pre_f == True and mix_layer4_feat is not None and (i_iters>8000 or i_iters<13) and fusio == True:
            # print('situation 1+')
            cf = self.decoder(cf, mix_layer4_feat, Mask, fusio)
        elif fusio == True and mix_layer4_feat is None:
            # print('situation 2+')
            cf, x_c = self.decoder(cf, mix_layer4_feat=None, Mask=None, fusio=True)
        else:
            cf = self.decoder(cf)
            # print('situation 3+')
        # print('cf.shape', cf.shape)

        if self.pre_f == True:
            with torch.no_grad():
                kf = self.model(kf)
                kf = self.decoder(kf)
                # print('kf.shape', kf.shape)
            if fusio == True:
                cf = interp_target(cf)
                # cf_feat = interp_target(cf_feat)
                # cf_aux = interp_target(cf_aux)
                kf = interp_target(kf)
                # kf_aux = interp_target(kf_aux)

            # kf_cpu = kf.cpu().numpy()
            interp_flow2cf = nn.Upsample(size=(kf.shape[-2], kf.shape[-1]), mode='bilinear', align_corners=True)
            interp_flow2cf_ratio = kf.shape[-2] / flow.shape[-2]
            # print('cf_aux.shape', cf_aux.shape)
            # print('kf_aux.shape', kf_aux.shape)
            flow_cf = (interp_flow2cf(flow) * interp_flow2cf_ratio).float().cuda(self.device)
            # print('flow_cf.shape', flow_cf.shape)
            # print('self.warp_bilinear(kf_aux, flow_cf).shape', self.warp_bilinear(kf_aux, flow_cf).shape)
            # print('torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1).shape', torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1).shape)

            # # 
            # # kf_aux_rec = np.zeros(cf_aux.shape)
            # kf_rec = np.zeros(cf.shape)
            # rec_positions = np.zeros(cf.shape)
            # for x in range(cf.shape[-1]):
            #     for y in range(cf.shape[-2]):
            #         x_flow = int(round(x - flow_cf[:, 0, y, x][0]))
            #         y_flow = int(round(y - flow_cf[:, 1, y, x][0]))
            #         if x_flow >= 0 and x_flow < flow_cf.shape[-1] and y_flow >= 0 and y_flow < flow_cf.shape[-2]:
            #             # kf_aux_rec[:, :, y_flow, x_flow] = kf_aux_cpu[:, :, y, x]
            #             kf_rec[:, :, y_flow, x_flow] = kf_cpu[:, :, y, x]
            #             rec_positions[:, :, y_flow, x_flow] = 1
            # # kf_aux_rec = torch.from_numpy(kf_aux_rec)
            # kf_rec = torch.from_numpy(kf_rec).float().cuda(self.device)
            # rec_positions = torch.from_numpy(rec_positions).float().cuda(self.device)


            if fusio == True:
                # print('cf.shape', cf.shape)
                # print('Masks_lt.shape', Masks_lt.shape)
                # print('rec_positions.shape', rec_positions.shape)
                # print('kf_rec.shape', kf_rec.shape)
                # pred_aux = self.sf_layer(torch.cat((cf_aux, (self.warp_bilinear(kf_aux, flow_cf)*(1-Masks_lt) + cf_aux*Masks_lt)), dim=1))
                # pred = self.decoder.sf_layer(torch.cat((cf, (rec_positions*kf_rec)*(1-Masks_lt)+cf*Masks_lt), dim=1))
                pred = self.decoder.sf_layer(torch.cat((cf, self.decoder.warp_bilinear(kf, flow_cf)*(1-Masks_lt)+cf*Masks_lt), dim=1))
            else:
                # pred_aux = self.sf_layer(torch.cat((cf_aux, self.warp_bilinear(kf_aux, flow_cf)), dim=1))
                # pred = decoder.sf_layer(torch.cat((cf, decoder.warp_bilinear(kf, flow_cf)), dim=1))
                # pred = self.decoder.sf_layer(torch.cat((cf, (rec_positions*kf_rec)), dim=1))
                pred = self.decoder.sf_layer(torch.cat((cf, self.decoder.warp_bilinear(kf, flow_cf)), dim=1))
            # del rec_positions, kf_rec, Masks_lt, kf_cpu, kf, flow_cf
            
            if fusio == True and mix_layer4_feat is None:
                return pred, x_c
            else:
                # del mix_layer4_feat
                return pred
            
        else:
            return cf


# def get_daformer(device):
#     model = DAFormer_tps(device = device)
#     return model








'''
import time

# ### compute model params

def count_parameters_pytorch(model):
    """
    计算模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    print('#### count_param ###')
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')

    from torch.autograd import Variable
    import numpy as np
    a = np.random.rand(1, 3, 512, 1024)
    a = np.array(a, dtype=np.float32)
    a =torch.tensor(a).to(device)
    x = torch.tensor(torch.rand(32,3,16,128,128),dtype=torch.float32).to(device)
    model = mit_b5().to(device)
    # model.init_weights()
    norm_cfg=dict(type='BN', requires_grad=True)
    decoder = DAFormerHead(num_classes = 11, decoder_params=dict(
                    embed_dims=256,
                    embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                    embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
                    fusion_cfg=dict(type='aspp',sep=True,dilations=(1, 6, 12, 18),pool=False,act_cfg=dict(type='ReLU'),norm_cfg=norm_cfg))).to(device)
    # param = count_param(model)
    # start = time.time()

    # print('a.shape', a.shape)
    # for i in range(1):
    #     y = model(a)
    #     out = decoder(y)
    #     print(i)
    # print("运行时间:%.2f秒"%(time.time()-start))
    # print('totoal parameters: %.2fM (%d)'%(param/1e6,param))
    # # print('y', y)
    # print('y[0].shape', y[0].shape) # [1, 64, 128, 128]
    # print('y[1].shape', y[1].shape) # [1, 128, 64, 128]
    # print('y[2].shape', y[2].shape) # [1, 320, 32, 64]
    # print('y[3].shape', y[3].shape) # [1, 512, 16, 32]
    # print('out.shape',   out.shape) # [1, 19, 128, 256]


    print(count_parameters_pytorch(model))
    print(count_parameters_pytorch(decoder))


    src_flow = np.random.rand(2, 512, 1024)
    src_flow = np.array(src_flow, dtype=np.float32)
    src_flow = torch.tensor(src_flow).unsqueeze(0).to(device)

    mix_feat = np.random.rand(1, 1024, 128, 256)
    mix_feat = np.array(mix_feat, dtype=np.float32)
    mix_feat = torch.tensor(mix_feat).to(device)


    Mask = np.random.rand(1, 512, 1024)
    Mask = np.array(Mask, dtype=np.float32)
    Mask = torch.tensor(Mask).to(device)

    from torch import nn
    interp_target = nn.Upsample(size=(512, 1024), mode='bilinear', align_corners=True)
    # start = time.time()
    # for i in range(10):
    #     out = DAFormer(a, kf = a, flow = src_flow, device = device, mix_layer4_feat = mix_feat, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, pre_f = True, fusio = True)
    #     # out = DAFormer(a, kf = a, flow = src_flow, device = device, mix_layer4_feat = None, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, pre_f = True, fusio = False)
    #     # out, x_c = DAFormer(a, kf = a, flow = src_flow, device = device, mix_layer4_feat = None, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, pre_f = True, fusio = True)
    #     # out = DAFormer(a, kf = a, flow = src_flow, device = device, mix_layer4_feat = None, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, pre_f = False, fusio = True)
    #     print(i)
    # print("time:%.2f"%(time.time()-start))

    


    daformer_model = DAFormer_tps(device = device, num_classes = 11)
    print(count_parameters_pytorch(daformer_model))


    # daformer_model = DAFormer_tps(device = device, pre_f = False)

    start = time.time()
    for i in range(30):
        # out = daformer_model(a, kf = a, flow = src_flow, mix_layer4_feat = mix_feat, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, fusio = True)
        out = daformer_model(a, kf = a, flow = src_flow, mix_layer4_feat = None, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, fusio = False)
        # out, x_c = daformer_model(a, kf = a, flow = src_flow, mix_layer4_feat = None, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, fusio = True)
        print('out.shape',   out.shape) # [1, 19, 128, 256]


        ####### out = daformer_model(a, kf = a, flow = src_flow, mix_layer4_feat = None, i_iters = 9000, Mask= Mask, Masks_lt= Mask, interp_target = interp_target, fusio = True)
        print(i)
    print("time:%.2f"%(time.time()-start))

'''





'''
    cfg = {
    'optimizer': {
        'type': 'AdamW',
        'lr': 6e-05,
        'betas': (0.9, 0.999),
        'weight_decay': 0.01,
        'paramwise_cfg': {
            'custom_keys': {
                'decoder': {'lr_mult': 10.0},
                'pos_block': {'decay_mult': 0.0},
                'norm': {'decay_mult': 0.0}
            }
        }
    },
    'lr_config': {
        'policy': 'poly',
        'warmup': 'linear',
        'warmup_iters': 1500,
        'warmup_ratio': 1e-6,
        'power': 1.0,
        'min_lr': 0.0,
        'by_epoch': False
    }
    }

    # 创建优化器
    paramwise_cfg = cfg['optimizer'].get('paramwise_cfg', {})
    custom_keys = paramwise_cfg.get('custom_keys', {})

    params = []

    for name, param in daformer_model.named_parameters():
        
        if not param.requires_grad:
            continue

        # if 'pos_block' in name:
        #     print('name_pos_block', name)
        # if 'decoder' in name:
        #     print('name_head', name)

        group = {'params': [param]}
        group['lr'] = cfg['optimizer']['lr']
        group['weight_decay'] = cfg['optimizer']['weight_decay']

        for key, value in custom_keys.items():
            if key in name:
                
                if 'lr_mult' in value:
                    # print('lr_mult in value')
                    # print('name', name)
                    group['lr'] = cfg['optimizer']['lr'] * value['lr_mult']
                if 'decay_mult' in value:
                    # print('decay_mult in value')
                    # print('name', name)
                    group['weight_decay'] = cfg['optimizer']['weight_decay'] * value['decay_mult']
                break
            
        params.append(group)




    optimizer = torch.optim.AdamW(params,
        lr=cfg['optimizer']['lr'],
        betas=cfg['optimizer']['betas'],
        weight_decay=cfg['optimizer']['weight_decay'])




    # 定义学习率调度策略
    def lr_lambda(current_step: int):
        lr_config = cfg['lr_config']
        warmup_iters = lr_config['warmup_iters']
        warmup_ratio = lr_config['warmup_ratio']
        min_lr = lr_config['min_lr']
        power = lr_config['power']
        max_iters = 40000  # 根据实际情况调整总的迭代次数

        if current_step < warmup_iters:
            lr = warmup_ratio + (cfg['optimizer']['lr'] - warmup_ratio) * (current_step / warmup_iters)
        else:
            progress = (current_step - warmup_iters) / (max_iters - warmup_iters)
            lr = cfg['optimizer']['lr'] * (1 - progress) ** power
        
        return max(lr, min_lr)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

'''


'''
conda activate /opt/software/anaconda3/envs/sepico
cd /home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS
python /home/customer/Desktop/ZZ/FMFSemi/TransferLearning/examples/domain_adaptation/video_tps/TPS/tps/model/mix_transformer2_tps.py

'''