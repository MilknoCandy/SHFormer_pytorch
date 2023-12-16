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

def featmap_visualize(model, output_file: str, data_generator: Dataset, img_size: tuple=(512, 512)):
    """绘制指定特征图

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
    # # 创建用于记录前向过程中的hook
    # def farward_hook(module, data_input, data_output):
    #     fmap_block.append(data_output)
    #     input_block.append(data_input)

    # # 由于transformer的输出都是序列, 需要重新reshape, 而从头到尾分别下采样了4/8/16/32倍
    # down_scale = {32:4, 64: 8, 160: 16, 256: 32}
    # 需要可视化的实例索引
    data_nums = len(data_generator)
    ind_select = random.sample(range(data_nums), 10)    # [List]
    ind_select = set(ind_select)                        # [Set]
    # ind_select = set([27,209])                        # [Set]
###########################创建画布以加速绘图######################################
    img_plot = np.zeros((img_size[0], img_size[1], 3))
    data_plot = np.zeros((9, img_size[0], img_size[1]))
    f = plt.figure(figsize=(12,8))
    ax1 = f.add_subplot(251, title="Image"); ax1.axis("off"); img_ori = ax1.imshow(img_plot)
    ax2 = f.add_subplot(252, title="mask1"); ax2.axis("off"); wt1 = ax2.imshow(data_plot[0], cmap="gray", vmin=0, vmax=1)
    ax3 = f.add_subplot(253, title="mask2"); ax3.axis("off"); wt2 = ax3.imshow(data_plot[1], cmap="gray", vmin=0, vmax=1)
    ax4 = f.add_subplot(254, title="mask3"); ax4.axis("off"); wt3 = ax4.imshow(data_plot[2], cmap="gray", vmin=0, vmax=1)
    ax5 = f.add_subplot(255, title="mask4"); ax5.axis("off"); wt4 = ax5.imshow(data_plot[3], cmap="gray", vmin=0, vmax=1)
    ax6 = f.add_subplot(256, title="groundtruth"); ax6.axis("off"); gt = ax6.imshow(data_plot[4], vmin=0, vmax=1, interpolation=None, cmap="gray")
    ax7 = f.add_subplot(257, title="mask_prob1"); ax7.axis("off"); wt1p = ax7.imshow(data_plot[5], vmin=0, vmax=1, cmap="jet")
    ax8 = f.add_subplot(258, title="mask_prob2"); ax8.axis("off"); wt2p = ax8.imshow(data_plot[6], vmin=0, vmax=1, cmap="jet")
    ax9 = f.add_subplot(259, title="mask_prob3"); ax9.axis("off"); wt3p = ax9.imshow(data_plot[7], vmin=0, vmax=1, cmap="jet")
    ax10 = f.add_subplot(2,5,10, title="mask_prob4"); ax10.axis("off"); wt4p = ax10.imshow(data_plot[8], vmin=0, vmax=1, cmap="jet")
    all_feats_figs = [wt1, wt2, wt3, wt4, wt1p, wt2p, wt3p, wt4p]
    plt.subplots_adjust(bottom=.05, top=.95, hspace=.1, wspace=.5)
    plt.tight_layout()

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
                input_gt = input_gt.cuda()
                # 由于imgo没有任何处理, 所有shape为(1, 192, 256, 3), 需要转换
                imgo_ = torch.clamp(imgo, min=0, max=255) / 255
                imgo_ = imgo_.permute(0, 3, 1, 2).cuda()
###########################获得网络中特征图部分####################################
            # 进行一次正向传播以获得所有输出
            output, mask_tokens, mask_prob_tokens = model(input_tensor)
###########################绘制网络中输出的特征图##################################
            # f = plt.figure()
            data_imgo = resize(imgo_, size=img_size, mode='bilinear'); data_imgo = data_imgo.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            data_gt = resize(input_gt, size=img_size, mode='bilinear'); data_gt = data_gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img_ori.set(data=data_imgo); gt.set(data=data_gt)

            for mask_id, (mask, mask_prob) in enumerate(zip(mask_tokens, mask_prob_tokens)):
                mask_ = mask.squeeze(); mask_prob_ = mask_prob.squeeze()
                data_mask = mask_.detach().cpu().numpy(); data_prob = mask_prob_.detach().cpu().numpy()
            #######################cv2绘图###########################
                # data = cv2.applyColorMap(np.uint8(data*255), cv2.COLORMAP_JET)    # 这里使用了和plt一样的颜色图（也可以用热力图JET）
                # cv2.imwrite(join(output_file, f"{i+1}_featmap_mask{mask_id}.jpg"), np.uint8(data*255))
                # matplotlib
                all_feats_figs[mask_id].set(data=data_mask); all_feats_figs[mask_id+4].set(data=data_prob)

            ####################绘制原图的uncertain#######################################

            #######################MatplotliB绘图###########################
            # plt.savefig(join(output_file, f"{i+1}_featmap_ori.jpg"))
            plt.savefig(join(output_file, f"{i+1}_featmap_ori.jpg"), bbox_inches='tight', pad_inches=0.1)
            #######################cv2绘图###########################
            # data = cv2.applyColorMap(np.uint8(data*255), cv2.COLORMAP_VIRIDIS)    # 这里使用了和plt一样的颜色图（也可以用热力图JET）
            # ori = cv2.cvtColor(np.uint8(255 * ori), cv2.COLOR_RGB2BGR)
            # add_img = cv2.addWeighted(ori, 0.2, data_, 0.8, dtype=cv2.CV_8U, gamma=0)
            # cv2.imwrite(join(output_file, f"{i+1}_uncertain_ori.jpg"), add_img)
            bar()
        print("done.")
