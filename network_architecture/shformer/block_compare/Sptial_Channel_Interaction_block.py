"""
❤Description: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-03-19 10:16:14
❤LastEditTime: 2023-03-19 10:16:16
❤Github: https://github.com/MilknoCandy
"""
from torch import nn
from functools import partial

class Fractorized_DWConv(nn.Module):
    def __init__(self, dim, bn=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim) if bn else None
        self.dwconv1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_ = x.transpose(1, 2).view(B, C, H, W)

        if self.bn is not None:
            x_ = self.bn(x_)

        x_ = self.dwconv1(x_)
        x_ = self.dwconv2(x_)
        x = x_.flatten(2).transpose(1, 2) + x
        return x

class Spatial_Channel_Interaction(nn.Module):
    def __init__(self, in_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Spatial_Channel_Interaction, self).__init__()
        self.channel_connect = nn.Sequential(
            norm_layer(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim//2)
        )
        self.spatial_connect = Fractorized_DWConv(dim=in_dim//2, bn=True)
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim//2),
            nn.GELU(),
            nn.Linear(in_dim//2, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x, H, W):
        x_ = self.channel_connect(x)
        x_ =self.spatial_connect(x_, H, W)
        sc_attn = self.proj(x_)
        # return sc_attn*x + x, sc_attn
        return sc_attn*x + x