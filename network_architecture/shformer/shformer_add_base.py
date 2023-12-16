"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-08 10:59:20
❤LastEditTime: 2023-12-16 21:08:36
❤Github: https://github.com/MilknoCandy
"""
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# from mmseg.models.builder import BACKBONES
# from mmseg.utils import get_root_logger
# from mmcv.runner import load_checkpoint
import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..shformer.mix_transformer_ori import mit_b0
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from network_architecture.simImple import resize
from ..shformer.decode_head_type import decode_head_sim
from network_architecture.network_tools.initilization import  InitWeights_He


class shformer_add_base(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, embed_dims=[32, 64, 160, 256], heads=[1, 2, 5, 8], linear_bias=True, mlp_ratio=4.,
        depths=[1, 1, 1, 1], mlp_sp=[16, 16, 4, 1], qkv_bias=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.,
        sr_ratios=[4,2,1,1], img_size=224, drop_rate=0., attn_drop_rate=0., qk_scale=None) -> None:
        super().__init__()
        ###################################################
        #               Encoder
        ###################################################
        self.backbone = mit_b0()
        ###################################################
        #               Decoder
        ###################################################
        self.decode_heade = decode_head_sim(embed_dims=embed_dims, norm_layer=norm_layer, num_classes=num_classes)
        self.apply(InitWeights_He())

    def forward(self, img):
        B, _, H, W = img.size()
        ###################################################
        #           ENCODER
        ###################################################
        feats = self.backbone(img)
        ###################################################
        #           DECODER
        ###################################################
        pred = self.decode_heade(feats)
        # return pred, mask_tokens, mask_prob_tokens      # for mask visualize
        return pred