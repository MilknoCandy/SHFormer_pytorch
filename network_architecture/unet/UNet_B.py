"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-03-30 11:05:38
❤LastEditTime: 2022-04-02 10:21:03
❤Github: https://github.com/MilknoCandy
"""

import torch
import torch.nn.functional as F
from torch import nn

class _EncoderBlock(nn.Module):
    """
    (conv => BN => Relu) * 2 => Maxpool 
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            # 用了padding, 保证输入和输出尺寸一致
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    """
    
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            # 这里也可以用nn.upsample的双线性插值来实现上采样, 原文用的是转置卷积
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        # 根据原文的数据集填写的
        self.enc1 = _EncoderBlock(in_channels, 64)        # 1-64
        self.enc2 = _EncoderBlock(64, 128)      # 64-128
        self.enc3 = _EncoderBlock(128, 256)     # 128-256
        self.enc4 = _EncoderBlock(256, 512, dropout=True)   # 256-512
        self.center = _DecoderBlock(512, 1024, 512)     # 512-1024-512, 这一步还未进行特征融合
        self.dec4 = _DecoderBlock(1024, 512, 256)       # 1024-512-256
        self.dec3 = _DecoderBlock(512, 256, 128)        # 512-256-128
        self.dec2 = _DecoderBlock(256, 128, 64)         # 256-128-64
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        # cat是长连接, 这里的interpolate作用是使enc4的h×w尺寸上采样为center的h×w一样, 当然也可以向jackcui代码里给的一样,用padding操作来还原图像大小
        # 在下采样的卷积过程中间部分我们保持图像尺寸不变, 因此长连接部分就不需要对原图像进行裁剪来改变尺寸进行特征融合
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        # 经过1×1卷积后再使用双线性插值将图片大小恢复原来尺寸
        return F.interpolate(final, x.size()[2:], mode='bilinear')