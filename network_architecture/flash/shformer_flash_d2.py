"""
❤Description: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-11-26 18:08:21
❤LastEditTime: 2023-12-16 21:06:35
❤FilePath: shformer_flash_d2
❤Github: https://github.com/MilknoCandy
"""
from functools import partial

from torch import nn

from network_architecture.network_tools.initilization import InitWeights_He

from ..shformer.decode_head_type import decode_head_best
from .flash_attn import flash_b1


class shformer_flash_d2(nn.Module):
    def __init__(self, num_classes=1, embed_dims=[32, 64, 160, 256],
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        ###################################################
        #               Encoder部分
        ###################################################
        self.backbone = flash_b1()
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