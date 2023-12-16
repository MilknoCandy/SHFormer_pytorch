"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-07 08:37:56
❤LastEditTime: 2022-12-07 08:37:57
❤Github: https://github.com/MilknoCandy
"""
from torch import nn
from timm.models.layers import trunc_normal_
import math

# from nnunet
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class InitWeights_XavierUniform(object):
    def __init__(self, gain=1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

class InitWeights(object):
    """
    各网络层的初始化
    He初始化
    """
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope
    
    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)     # 和LayerNorm的初始化一样，偏差设为零，权重为壹
            nn.init.constant_(module.bias, 0)

        # if isinstance(module, nn.Linear) and module.bias is not None:
        #     module.bias.data.zero_()

        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)

        # if isinstance(module, nn.Conv2d):
        #     '''
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        #     trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        #     if m.bias is not None:
        #         nn.init.zeros_(m.bias)
        #     '''
        #     nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        #     if m.bias is not None:
        #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        #         bound = 1 / math.sqrt(fan_in)
        #         nn.init.uniform_(module.bias, -bound, bound)
            

