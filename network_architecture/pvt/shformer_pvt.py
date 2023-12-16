"""
❤Description: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-11-27 21:23:46
❤LastEditTime: 2023-12-16 21:09:07
❤FilePath: shformer_pvt
❤Github: https://github.com/MilknoCandy
"""
import math
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mix_pvt import pvt_b0
from ..shformer.decode_head_type import *
from network_architecture.network_tools.initilization import  InitWeights_He

class shformer_pvt(nn.Module):
    def __init__(self, num_classes=1, embed_dims=[32, 64, 160, 256], norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        ###################################################
        #               Encoder
        ###################################################
        self.backbone = pvt_b0()
        ###################################################
        #               Decoder
        ###################################################
        self.decode_heade = decode_head_best(embed_dims=embed_dims, norm_layer=norm_layer, num_classes=num_classes)
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
        return pred