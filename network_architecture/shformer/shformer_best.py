"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-08 20:51:16
❤LastEditTime: 2023-12-16 21:08:14
❤Github: https://github.com/MilknoCandy
"""
from functools import partial

import torch
from network_architecture.network_tools.initilization import InitWeights_He
from network_architecture.simImple import resize
from torch import nn

from ..shformer.mix_transformer import mit_b0


class Flat_linear(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, norm_layer=None, is_act=False, act_layer=None, drop=0.,
        re_construct=False):
        super().__init__()
        out_features = out_features or in_features
        self.proj = nn.Linear(in_features, out_features//2)
        self.proj2 = nn.Linear(out_features//2, out_features)
        self.norm = None
        self.norm2 = None
        if norm_layer is not None:
            self.norm = norm_layer(out_features//2)
            self.norm2 = norm_layer(out_features)
        self.is_act = is_act
        if self.is_act:
            self.act = act_layer()
        self.re_construct = re_construct
        
    def forward(self, x):
        B, _, H, W = x.size()
        x = x.flatten(2).transpose(1,2)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.is_act:
            x = self.act(x)
        x = self.proj2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        if self.re_construct:
            return x.transpose(1,2).reshape(B, -1, H, W)
        else:
            return x

class shformer_best(nn.Module):
    def __init__(self, num_classes=1, embed_dims=[32, 64, 160, 256], norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        ###################################################
        #               Encoder
        ###################################################
        self.backbone = mit_b0()
        ###################################################
        #               Decoder
        ###################################################
        self.linear_decoder3 = Flat_linear(
            in_features=embed_dims[3], out_features=embed_dims[2], norm_layer=norm_layer, act_layer=nn.GELU, is_act=False, re_construct=True)
        self.conv_decoder3 = nn.Conv2d(embed_dims[2], embed_dims[2], 3, 1, 1, groups=embed_dims[2])

        self.linear_decoder2 = Flat_linear(
            in_features=embed_dims[2], out_features=embed_dims[1], norm_layer=norm_layer, act_layer=nn.GELU, is_act=False, re_construct=True)
        self.conv_decoder2 = nn.Conv2d(embed_dims[1], embed_dims[1], 3, 1, 1, groups=embed_dims[1])
    
        self.linear_decoder1 = Flat_linear(
            in_features=embed_dims[1], out_features=embed_dims[0], norm_layer=norm_layer, act_layer=nn.GELU, is_act=False, re_construct=True)
        self.conv_decoder1 = nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1, groups=embed_dims[0])
        self.linear_pred = nn.Conv2d(embed_dims[0], num_classes, 3, padding=1)
        self.apply(InitWeights_He())
    
    def forward(self, img):
        ###################################################
        #           ENCODER
        ###################################################
        c1, c2, c3, c4 = self.backbone(img)
        ###################################################
        #           DECODER
        ###################################################
        _c4 = self.linear_decoder3(c4)
        _c4 = resize(_c4, scale_factor=2, mode='bilinear')

        _c3 = torch.add(c3, _c4)
        _c3 = self.conv_decoder3(_c3) + _c3

        _c3 = self.linear_decoder2(_c3)
        _c3 = resize(_c3, scale_factor=2, mode='bilinear')

        _c2 = torch.add(c2, _c3)
        _c2 = self.conv_decoder2(_c2) + _c2

        _c2 = self.linear_decoder1(_c2)
        _c2 = resize(_c2, scale_factor=2, mode='bilinear')

        _c1 = torch.add(c1, _c2)
        _c1 = self.conv_decoder1(_c1) + _c1
        
        pred = self.linear_pred(_c1)
        return resize(pred, scale_factor=4, mode='bilinear')