"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-02 16:20:15
❤LastEditTime: 2022-11-23 15:28:03
❤Github: https://github.com/MilknoCandy
"""
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from alive_progress import alive_bar
from batchgenerators.utilities.file_and_folder_operations import *
from torch import nn
from torch.utils.data import Dataset
from utils.dict_tools.dict_combine import combine_deepest_dict
from network_architecture.simImple import resize

def np_softmax(feats):
    """
    Args:
        feats: HxW
    """
    return np.exp(feats) / np.sum(np.exp(feats))

def get_uncertainty_map_spatial(feature):
    """在不同分布中计算信息熵然后加和并在空间维度进行归一化得到空间中每个像素的权重
        feature(Tensor): BxCxHxW
    """
    B, _, H, W = feature.size()
    # 将不同通道视作不同分布
    uncertainty_spatial = nn.Softmax(dim=1)(feature)     # BxCxHxW
    eps = 1e-20
    top = uncertainty_spatial * torch.log(uncertainty_spatial + eps)
    bottom = torch.log(torch.Tensor([uncertainty_spatial.size()[1]])).cuda()
    entropy = (- top / bottom).sum(dim=1)   # BxL
    attn = 1 - entropy.squeeze().reshape(H, W)   # 由于固定B为1, 因此只需要图像分辨率
    # attn = 1 - entropy.reshape(B, H, W).unsqueeze(1)
    return attn

def entropy_visualize(model, output_file: str, data_generator: Dataset, img_size: tuple=(512, 512)):
    """绘制指定层输出的特征图的根据信息熵计算得到的uncertainty的特征图

    Args:
        model (_type_): _description_
        output_file (str): _description_
        data_generator (Dataset): _description_
        img_size (tuple, optional): _description_. Defaults to (512, 512).
    """
    # 可视化原图像的不确定像素
    # 创建不确定可视化存放文件夹
    maybe_mkdir_p(output_file)
    model.eval()
    # 创建用于记录前向过程中的hook
    def farward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # 由于transformer的输出都是序列, 需要重新reshape, 而从头到尾分别下采样了4/8/16/32倍
    down_scale = {32:4, 64: 8, 160: 16, 256: 32}
    # 需要可视化的实例索引
    data_nums = len(data_generator)
    ind_select = random.sample(range(data_nums), 10)    # [List]
    ind_select = set(ind_select)                        # [Set]
    # ind_select = set([27,209])                        # [Set]
    # 选择需要可视化的层
    # 最深层的值为该层输出的特征图的名字
    layers_select_with_name = {
            "backbone": {
                "norm1": "sa1",
                "norm2": "sa2",
                "norm3": "sa3",
                "norm4": "sa4",
            }
                }
    layers = combine_deepest_dict(layers_select_with_name, [])
    # 循环获取网络层的hook
    feat_names = list()
    # 获取网络层不能在案例循环中, 这样会每次多获取一次, 所以receptive那个函数里先获取网络层再在循环中hook
    for layer_and_ftname in layers:
        # 将层的名字和特征图的名字分开
        layer, ftname = layer_and_ftname.rsplit('.', 1)
        feat_names.append(ftname)   # 添加到列表供绘图使用
        # 对指定层获取特征图(如果为Sequential则会直接获得最终输出图, 
        # 若为ModuleList则需要获取指定层, 当然通常为最后一层)
        layer_ = model.get_submodule(layer)
        if isinstance(layer_, nn.ModuleList):
            # layer_ = layer_[-1] = 会多获取一个, 因为相当于两次引用
            layer_[-1].register_forward_hook(farward_hook)
        else:
            layer_.register_forward_hook(farward_hook)

    # 遍历测试集图像
    print("="*20, f"output Testing uncertainty_featmaps", "="*20)
    with alive_bar(data_nums, spinner='twirl', length=80, force_tty=True, title="Output Testing featmaps") as bar:  
        for i, data in enumerate(data_generator):
            if i not in ind_select:
                bar()
                continue
            
            input_tensor, input_gt, _, imgo = data

            if torch.cuda.is_available():
                model = model.cuda()
                input_tensor = input_tensor.cuda()
                # 由于imgo没有任何处理, 所有shape为(1, 192, 256, 3), 需要转换
                imgo_ = torch.clamp(imgo, min=0, max=255) / 255
                imgo_ = imgo_.permute(0, 3, 1, 2).cuda()
###########################获得网络中特征图部分##################################
            fmap_block = list()
            input_block = list()

            # 进行一次正向传播以获得所有输出
            output = model(input_tensor)

            for feature_map, ftname in zip(fmap_block, feat_names):

                if len(feature_map.shape) < 4:
                    stage_scale = down_scale[feature_map.shape[2]]
                    feature_map = feature_map.reshape(1, img_size[0]//stage_scale, img_size[1]//stage_scale, -1).permute(0, 3, 1, 2).contiguous()
                    # 特征图的channel维度和batch维度去掉，得到二维张量
                    # 等于4说明是二维矩阵, 前两个维度分别为batch和channel维度
###########################获得网络中特征图的不确定部分##################################
                uncertainty_map = get_uncertainty_map_spatial(feature_map)
                uncertainty_map = resize(uncertainty_map[None, None,...], size=img_size, mode='nearest')[0,0]
                # 不放缩至图像尺寸, 这样更能看出注意力
                data_ = uncertainty_map.detach().cpu().numpy()
                # data = (data_ - data_.min()) / (data_.max() - data_.min() + 1e-9)
                # data = 
                #######################cv2绘图###########################
                data = cv2.applyColorMap(np.uint8(data_*255), cv2.COLORMAP_VIRIDIS)    # 这里使用了和plt一样的颜色图（也可以用热力图JET）
                cv2.imwrite(join(output_file, f"{i+1}_uncertain_{ftname}.jpg"), data)
            ####################绘制原图的uncertain#######################################
            uncertainty_map = get_uncertainty_map_spatial(imgo_)
            # 不放缩至图像尺寸, 这样更能看出注意力
            data_ = uncertainty_map.detach().cpu().numpy()
            # data = (data_ - data_.min()) / (data_.max() - data_.min() + 1e-9)
            ori = imgo_[0,...].permute(1,2,0).cpu().numpy()
            mask_ = input_gt.squeeze().cpu().numpy()
            uncertain_prob = np_softmax(data_/data_.min())
            mask_uncertain = 1.0*(uncertain_prob>=0.5)
            #######################MatplotliB绘图###########################
            plt.figure()
            plt.subplot(221)
            plt.axis('off')
            plt.imshow(ori)
            plt.subplot(222)
            plt.axis('off')
            plt.imshow(ori)
            plt.imshow(data_, cmap='viridis', alpha=0.6)
            plt.subplot(223)
            plt.axis('off')
            plt.imshow(mask_, cmap='gray')
            plt.subplot(224)
            plt.axis('off')
            plt.imshow(mask_uncertain, cmap='gray')
            plt.savefig(join(output_file, f"{i+1}_uncertain_ori.jpg"))
            #######################cv2绘图###########################
            # data = cv2.applyColorMap(np.uint8(data*255), cv2.COLORMAP_VIRIDIS)    # 这里使用了和plt一样的颜色图（也可以用热力图JET）
            # ori = cv2.cvtColor(np.uint8(255 * ori), cv2.COLOR_RGB2BGR)
            # add_img = cv2.addWeighted(ori, 0.2, data_, 0.8, dtype=cv2.CV_8U, gamma=0)
            # cv2.imwrite(join(output_file, f"{i+1}_uncertain_ori.jpg"), add_img)
            bar()
        print("done.")
