"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-08 10:59:49
❤LastEditTime: 2023-12-02 15:25:40
❤Github: https://github.com/MilknoCandy
"""
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from network_architecture.simImple import resize
from torch import nn

__all__ = [
    "decode_head_best",
    "decode_head_test",
    "decode_head_PLD",
    "decode_head_MLA"
]

    
class Feature_Enhance(nn.Module):
    def __init__(self, in_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        ###################Branch one###################
        self.ln1 = norm_layer(in_dim)
        self.act1 = nn.GELU()
        self.linear1 = nn.Linear(in_dim, in_dim)    # 考虑多个Linear或拆成多组分别linear（类似self-attention）
        ###################Branch two###################
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim)
    
        # self.bn4all = nn.BatchNorm2d(in_dim)  # 不用softmax得到概率值，而是类似卷积或Linaer去学习到每个特征点的权值，然后用BN来稳定
    def forward(self, x, H, W):
        """插在Transformer Encoder的MLP后面

        Args:
            x (Tensor): BxLxD
        """
        eps = 1e-20
        B, L, D = x.shape
        # x = x.transpose(1,2).reshape(B, -1, H, W)
        # x_c = self.ln1(x.flatten(2).transpose(1,2))     # B×L×D
        x_c = self.ln1(x)     # B×L×D
        x_c = self.act1(x_c)
        x_c = self.linear1(x_c)
        x = x.transpose(1,2).reshape(B, -1, H, W)
        # for atto+sci
        # attn_channel = x_c.transpose(1,2).reshape(B, -1, H, W)
        attn_channel = nn.Softmax(dim=1)(x_c)
        attn_channel = attn_channel.transpose(1,2).reshape(B, -1, H, W)

        x_s = self.bn1(x)
        x_s = self.relu(x_s)
        x_s = self.conv1(x_s)
        attn_spatial = nn.Softmax(dim=1)(x_s)
        attn = attn_channel * attn_spatial
        # x_attn = self.bn4all(x*attn)
        # x_ = x + x_attn
        x_sc = x*attn
        x_ = x + x_sc
        # x_ = x*attn
        return x_.flatten(2).transpose(1,2)
        # return x_.flatten(2).transpose(1,2), attn

"""BEST decoder"""
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

class Conv_linear(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, norm_layer=None, is_act=False, act_layer=None, drop=0.,
        re_construct=False):
        super().__init__()
        out_features = out_features or in_features
        self.proj = nn.Conv2d(in_features, out_features//2, 1)
        self.proj2 = nn.Conv2d(out_features//2, out_features, 1)
        self.norm = nn.BatchNorm2d(out_features//2)
        self.norm2 = nn.BatchNorm2d(out_features)
        self.is_act = is_act
        if self.is_act:
            self.act = act_layer()
        self.re_construct = re_construct
        
    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.is_act:
            x = self.act(x)
        x = self.proj2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        return x


class decode_head_test(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512], norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1) -> None:
        super().__init__()
        self.linear_decoder3 = nn.Sequential(
            ConvModule(in_channels=embed_dims[3], out_channels=embed_dims[2]//2, kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embed_dims[2]//2, out_channels=embed_dims[2], kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True))
        )
        self.linear_decoder2 = nn.Sequential(
            ConvModule(in_channels=embed_dims[2], out_channels=embed_dims[1]//2, kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embed_dims[1]//2, out_channels=embed_dims[1], kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True))
        )
        self.linear_decoder1 = nn.Sequential(
            ConvModule(in_channels=embed_dims[1], out_channels=embed_dims[0]//2, kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True)),
            ConvModule(in_channels=embed_dims[0]//2, out_channels=embed_dims[0], kernel_size=1, norm_cfg=dict(type='BN', requires_grad=True))
        )
        self.linear_pred = nn.Conv2d(embed_dims[0], num_classes, 3, padding=1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = self.linear_decoder3(c4)
        _c4 = resize(_c4, scale_factor=2, mode='bilinear')

        _c3 = torch.add(c3, _c4)

        _c3 = self.linear_decoder2(_c3)
        _c3 = resize(_c3, scale_factor=2, mode='bilinear')

        _c2 = torch.add(c2, _c3)

        _c2 = self.linear_decoder1(_c2)
        _c2 = resize(_c2, scale_factor=2, mode='bilinear')

        _c1 = torch.add(c1, _c2)
        
        pred = self.linear_pred(_c1)
        return resize(pred, scale_factor=4, mode='bilinear')
    
class decode_head_best(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512], norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1) -> None:
        super().__init__()
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

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

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

"""Linear + Conv + Linear"""
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Sandwich_Module(nn.Module):
    def __init__(self, in_features, out_features=None, norm_layer=None, act_layer=None, is_norm=False, is_act=False, drop=0., re_construct=False) -> None:
        super().__init__()
        out_features = out_features or in_features
        """深度可分离卷积的思想"""
        self.proj1 = nn.Linear(in_features, out_features//2)
        self.norm1 = norm_layer(out_features//2) if norm_layer is not None else None
        self.dwconv = DWConv(out_features//2)
        self.act = act_layer if act_layer is not None else None
        self.proj2 = nn.Linear(out_features//2, out_features)
        self.re_construct = re_construct
    
    def forward(self, x):
        B, _, H, W = x.shape
        # 单分支
        x = x.flatten(2).transpose(1,2)
        x = self.proj1(x)
        if self.norm1 is not None:
            x = self.norm1(x)

        x = self.dwconv(x, H, W)

        x = self.proj2(x)
        if self.re_construct:
            return x.transpose(1,2).reshape(B, -1, H, W)
        else:
            return x

class decode_head_sand(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512], norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1) -> None:
        super().__init__()
        self.linear_decoder3 = Sandwich_Module(
            in_features=embed_dims[3], out_features=embed_dims[2],
            norm_layer=norm_layer, act_layer=nn.GELU, 
            re_construct=True
        )

        self.linear_decoder2 = Sandwich_Module(
            in_features=embed_dims[2], out_features=embed_dims[1],
            norm_layer=norm_layer, act_layer=nn.GELU, 
            re_construct=True
        )
    
        self.linear_decoder1 = Sandwich_Module(
            in_features=embed_dims[1], out_features=embed_dims[0],
            norm_layer=norm_layer, act_layer=nn.GELU, 
            re_construct=True
        )
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embed_dims[0], num_classes, 1, padding=0)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        _c4 = self.linear_decoder3(c4)
        _c4 = resize(_c4, scale_factor=2, mode='bilinear')

        _c3 = torch.add(c3, _c4)
        _c3 = self.linear_decoder2(_c3)
        _c3 = resize(_c3, scale_factor=2, mode='bilinear')

        _c2 = torch.add(c2, _c3)
        _c2 = self.linear_decoder1(_c2)
        _c2 = resize(_c2, scale_factor=2, mode='bilinear')
        
        _c1 = torch.add(c1, _c2)
        x = self.dropout(_c1)
        x = self.linear_pred(x)
        return resize(x, scale_factor=4, mode='bilinear')

class decode_head_sim(nn.Module):
    def __init__(self, embed_dims=[64, 128, 320, 512], norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1) -> None:
        super().__init__()
        self.linear_decoder3 = Flat_linear(
            in_features=embed_dims[3], out_features=embed_dims[2], norm_layer=None, act_layer=nn.GELU, is_act=True, re_construct=True)
        self.conv_decoder3 = nn.Conv2d(embed_dims[2], embed_dims[2], 3, 1, 1, groups=embed_dims[2])

        self.linear_decoder2 = Flat_linear(
            in_features=embed_dims[2], out_features=embed_dims[1], norm_layer=None, act_layer=nn.GELU, is_act=True, re_construct=True)
        self.conv_decoder2 = nn.Conv2d(embed_dims[1], embed_dims[1], 3, 1, 1, groups=embed_dims[1])
    
        self.linear_decoder1 = Flat_linear(
            in_features=embed_dims[1], out_features=embed_dims[0], norm_layer=None, act_layer=nn.GELU, is_act=True, re_construct=True)
        self.conv_decoder1 = nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1, groups=embed_dims[0])
        self.dropout = nn.Dropout(0.1)
        self.linear_pred = nn.Conv2d(embed_dims[0], num_classes, 1, padding=0)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

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
        
        # x = self.dropout(_c1)
        pred = self.linear_pred(_c1)
        return resize(pred, scale_factor=4, mode='bilinear')

"""ssformer's decoder: PLD"""
class conv(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=512, embed_dim=768, k_s=3):
        super().__init__()

        self.proj = nn.Sequential(nn.Conv2d(input_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False), nn.ReLU())

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class decode_head_PLD(nn.Module):
    def __init__(self, dims, dim, class_num=2):
        super(decode_head_PLD, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        L34 = self.linear_fuse34(torch.cat([_c4, _c3], dim=1))
        L2 = self.linear_fuse2(torch.cat([L34, _c2], dim=1))
        _c = self.linear_fuse1(torch.cat([L2, _c1], dim=1))


        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

"""SETR's decoder: MLA"""
class decode_head_MLA(nn.Module):

    def __init__(self, dims, dim, class_num=2):
        super(decode_head_MLA, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = conv(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = conv(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = conv(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = conv(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_pred = nn.Conv2d(4 * embedding_dim,self.num_classes, 3, padding=1)

    def forward(self, inputs):

        c1, c2, c3, c4 = inputs
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=True)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=True)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=True)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        
        _c = self.linear_pred(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        return _c

"""Saegformer's decoder"""
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=512, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class decode_head_SeD(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, dims, dim, class_num=2):
        super(decode_head_SeD, self).__init__()
        self.num_classes = class_num

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = dims[0], dims[1], dims[2], dims[3]
        embedding_dim = dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse34 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse2 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))
        self.linear_fuse1 = ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,norm_cfg=dict(type='BN', requires_grad=True))

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):

        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x