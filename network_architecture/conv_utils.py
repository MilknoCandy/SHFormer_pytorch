"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-07 08:39:28
❤LastEditTime: 2022-12-07 08:39:29
❤Github: https://github.com/MilknoCandy
"""
from torch import nn


class Conv(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, stride=1, bn=False, relu=True, bias=True, padding=None, groups=1) -> None:
        super().__init__()
        self.in_feat = in_feat
        if padding is None:
            padding=(kernel_size-1)//2
        self.conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding=padding, bias=bias, groups=groups)
        self.bn = None
        self.relu = None
        if relu:
            # self.relu = nn.ReLU(inplace=True)
            self.relu = nn.LeakyReLU(inplace=True)
            # self.relu = nn.PReLU(num_parameters=1, init=0.25)
        
        if bn:
            self.bn = nn.BatchNorm2d(out_feat)
        
        
    def forward(self, x):
        assert x.size()[1] == self.in_feat, f"{x.size()[1]} {self.in_feat}"
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x