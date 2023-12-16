"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-07 08:38:29
❤LastEditTime: 2022-12-08 10:23:57
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
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from network_architecture.conv_utils import Conv
# from network_architecture.shformer.decode_head_type import Feature_Enhance
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    "mit_b0",
    "mit_b1",
    "mit_b2",
]
class Feature_Enhance(nn.Module):
    def __init__(self, in_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6)) -> None:
        super().__init__()
        ###################Branch one###################
        self.ln1 = norm_layer(in_dim)
        self.act1 = nn.GELU()
        self.linear1 = nn.Linear(in_dim, in_dim)
        ###################Branch two###################
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim)

        self.norm = nn.BatchNorm2d(in_dim)
    
    def forward(self, x, H, W):
        """插在Transformer Encoder的MLP后面

        Args:
            x (Tensor): BxLxD
        """
        eps = 1e-20
        B, L, D = x.shape
        x_c = self.ln1(x)     # B×L×D
        x_c = self.act1(x_c)
        x_c = self.linear1(x_c)
        attn_channel = nn.Softmax(dim=1)(x_c)
        attn_channel = attn_channel.transpose(1,2).reshape(B, -1, H, W)

        x = x.transpose(1,2).reshape(B, -1, H, W)
        x_s = self.bn1(x)
        x_s = self.relu(x_s)
        x_s = self.conv1(x_s)
        attn_spatial = nn.Softmax(dim=1)(x_s)
        attn = attn_channel * attn_spatial
        x_ = x + self.norm(x*attn)                                 # B×D×H×W
        return x_.flatten(2).transpose(1,2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        # self.fe = Feature_Enhance(in_dim=out_features)      # 额外的特征增强
        self.apply(self._init_weights)

    def remove_cls(self, x):
        """ Remove weight token. """
        return x[:, 1:, :], x[:, :1]

    def insert_cls(self, x, wt_token):
        """ Insert weight token. """
        x = torch.cat((wt_token, x), dim=1)
        return x

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x, wt_token = self.remove_cls(x)
        x = self.dwconv(x, H, W)
        x = self.insert_cls(x, wt_token)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x, wt_token = self.remove_cls(x)
        # x = self.fe(x, H, W)
        # x = self.drop(x)
        # x = self.insert_cls(x, wt_token)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # assert num_heads in [1, 2, 5, 8], "num_heads should in [1, 2, 5, 8]. This is designed for ConvRelPosEnc"
        # if num_heads==1:
        #     window=3
        # elif num_heads==2:
        #     window={3: 1, 5:1}
        # elif num_heads==5:
        #     window={3: 2, 5: 2, 7:1}
        # else:
        #     window={3:2, 5:3, 7:3}

        # self.crpe = ConvRelPosEnc(dim // num_heads, h=num_heads, window=window)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 降维以加速
            # if sr_ratio==8:
            #     # 降维以加速
            #     # self.sr = nn.Sequential(
            #     #     Conv(dim, dim//4, kernel_size=sr_ratio//4, stride=sr_ratio//4, bn=True, relu=True, padding=0),
            #     #     Conv(dim//4, dim//2, kernel_size=sr_ratio//4, stride=sr_ratio//4, bn=True, relu=True, padding=0),
            #     #     Conv(dim//2, dim, kernel_size=sr_ratio//4, stride=sr_ratio//4, bn=False, relu=False, padding=0)
            #     # )
            #     self.sr = nn.Sequential(
            #         nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            #     )

            # elif sr_ratio==4:
            #     # self.sr = nn.Sequential(
            #     #     Conv(dim, dim//2, kernel_size=sr_ratio//2, stride=sr_ratio//2, bn=True, relu=True, padding=0),
            #     #     Conv(dim//2, dim, kernel_size=sr_ratio//2, stride=sr_ratio//2, bn=False, relu=False, padding=0)
            #     # )
            #     self.sr = nn.Sequential(
            #         nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            #     )
            # else:
            #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            # self.sr_st = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
            # self.norm_act = nn.Sequential(
            #     nn.LayerNorm(dim),
            #     nn.GELU())

        self.apply(self._init_weights)

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

    def remove_cls(self, x):
        """ Remove weight token. """
        return x[:, 1:, :], x[:, :1]

    def insert_cls(self, x, wt_token):
        """ Insert weight token. """
        x = torch.cat((wt_token, x), dim=1)
        return x

    def get_featmap_for_select(self, feat, wt_token):
        mask_weight = feat @ wt_token.transpose(1,2)
        mask_prob = mask_weight.sigmoid()
        mask = 1.0*(mask_prob>0.5)
        # 检查是否每张图像都有权重大于0.5的特征点, 若没有则选择全部点
        mask_count = mask.sum(dim=(1,2))
        if (mask_count==0).sum() != 0:
            mask[mask_count==0, ...] = 1    # =0说明未出现大于0.5的点, 则选择全部
        return mask, mask_prob, mask_weight

    def simFeats_fusion(self, x, mask, weight, H, W):
        """将特征根据mask进行分离并分别进行融合(这样实现大范围的token联系)然后沿序列维度Cat"""
        B, L, D = x.shape
        x_po = x*mask
        weight_po = weight*mask
        x_ne = x*(1-mask)
        weight_ne = weight*(1-mask)
        # x_po = x_po.permute(0, 2, 1).reshape(B, D, H, W)
        # x_ne = x_ne.permute(0, 2, 1).reshape(B, D, H, W)
        # x_po_f = self.sr(x_po)
        # x_ne_f = self.sr(x_ne)
        x_po_f = (x_po * weight_po).sum(dim=1, keepdim=True)
        x_ne_f = (x_ne * weight_ne).sum(dim=1, keepdim=True)
        # x_ = torch.cat([x_po_f, x_ne_f], dim=2).reshape(B, D, -1).permute(0, 2, 1)
        x_ = torch.cat([x_po_f, x_ne_f], dim=1)
        # x_ = self.norm(x_)
        return x_


    def forward(self, x, H, W):
        B, N, C = x.shape

        x, wt_token = self.remove_cls(x)
        mask, mask_prob, mask_weight = self.get_featmap_for_select(x, wt_token)
        # x_mask_ = x*mask
        # x_mask = self.insert_cls(x_mask_, wt_token)
        x = self.insert_cls(x, wt_token)

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q = x.reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(0)
        if self.sr_ratio > 1:
            # 在计算query的时候已经取出wt_token, 因此这里直接用
            x_, wt_token = self.remove_cls(x)
            # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 完成分离+融合操作
            x_ = self.simFeats_fusion(x_, mask, mask_weight, H, W)
            # x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)

            x_ = self.insert_cls(x_, wt_token)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = attn @ v
        # x = self.crpe(x, H, W).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, mask, mask_prob


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # feature enhance
        # self.fe = Feature_Enhance(in_dim=dim)
        # self.fe_norm = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

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

    def forward(self, x, H, W):
        x_, mask, mask_prob = self.attn(self.norm1(x), H, W)
        x = x + self.drop_path(x_)
        # x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x, mask, mask_prob


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # self.img_size = img_size
        self.patch_size = patch_size
        # self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.H, self.W = 192 // patch_size[0], 256 // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # Class tokens.
        self.wt_token1 = nn.Parameter(torch.zeros(1, 1, embed_dims[0]))
        self.wt_token2 = nn.Parameter(torch.zeros(1, 1, embed_dims[1]))
        self.wt_token3 = nn.Parameter(torch.zeros(1, 1, embed_dims[2]))
        self.wt_token4 = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # self.bound1 = Boundary_attention(in_dim=embed_dims[0], norm_layer=norm_layer)
        # self.bound2 = Boundary_attention(in_dim=embed_dims[1], norm_layer=norm_layer)
        # self.bound3 = Boundary_attention(in_dim=embed_dims[2], norm_layer=norm_layer)
        # self.bound4 = Boundary_attention(in_dim=embed_dims[3], norm_layer=norm_layer)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights.
        trunc_normal_(self.wt_token1, std=.02)
        trunc_normal_(self.wt_token2, std=.02)
        trunc_normal_(self.wt_token3, std=.02)
        trunc_normal_(self.wt_token4, std=.02)
        self.apply(self._init_weights)

    def insert_cls(self, x, wt_token):
        """ Insert weight token. """
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        wt_token = wt_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((wt_token, x), dim=1)
        return x

    def remove_cls(self, x):
        """ Remove CLS token. """
        return x[:, 1:, :], x[:, :1]

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

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
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
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        wt_tokens = []
        mask_tokens = []
        mask_prob_tokens = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        x = self.insert_cls(x, self.wt_token1)
        for i, blk in enumerate(self.block1):
            x, mask, mask_prob = blk(x, H, W)
        x = self.norm1(x)
        x, wt_token = self.remove_cls(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask_prob = mask_prob.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.bound1(x)
        outs.append(x)
        wt_tokens.append(wt_token)
        mask_tokens.append(mask)
        mask_prob_tokens.append(mask_prob)

        # stage 2
        x, H, W = self.patch_embed2(x)
        x = self.insert_cls(x, self.wt_token2)
        for i, blk in enumerate(self.block2):
            x, mask, mask_prob = blk(x, H, W)
        x = self.norm2(x)
        x, wt_token = self.remove_cls(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask_prob = mask_prob.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.bound2(x)
        outs.append(x)
        wt_tokens.append(wt_token)
        mask_tokens.append(mask)
        mask_prob_tokens.append(mask_prob)

        # stage 3
        x, H, W = self.patch_embed3(x)
        x = self.insert_cls(x, self.wt_token3)
        for i, blk in enumerate(self.block3):
            x, mask, mask_prob = blk(x, H, W)
        x = self.norm3(x)
        x, wt_token = self.remove_cls(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask_prob = mask_prob.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.bound3(x)
        outs.append(x)
        wt_tokens.append(wt_token)
        mask_tokens.append(mask)
        mask_prob_tokens.append(mask_prob)

        # stage 4
        x, H, W = self.patch_embed4(x)
        x = self.insert_cls(x, self.wt_token4)
        for i, blk in enumerate(self.block4):
            x, mask, mask_prob = blk(x, H, W)
        x = self.norm4(x)
        x, wt_token = self.remove_cls(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask = mask.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        mask_prob = mask_prob.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # x = self.bound4(x)
        outs.append(x)
        wt_tokens.append(wt_token)
        mask_tokens.append(mask)
        mask_prob_tokens.append(mask_prob)

        # return outs, wt_tokens
        return outs, mask_tokens, mask_prob_tokens

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
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



# @BACKBONES.register_module()
class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1, 1, 1], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b3(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b3, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b4(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b4, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b5(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b5, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)