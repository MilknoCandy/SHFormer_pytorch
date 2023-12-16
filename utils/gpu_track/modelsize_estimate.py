"""
❤Descripttion: This code is used to compute model params and FLOPs manually, but there are still bugs now!
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2023-02-21 11:09:51
❤LastEditTime: 2023-12-03 15:31:43
❤Github: https://github.com/MilknoCandy
"""
import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = now_dir.rsplit('/', 2)[0]
sys.path.append(root_dir)
import numpy as np
import torch
import torch.nn as nn

from network_architecture.network_tools.build_model import build_network


def modelsize(model, input_size, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # print('Model {} : Number of params: {}'.format(model._get_name(), para))
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input = torch.randn(*input_size)
    input_ = input.clone().cuda()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    # print('Model {} : Number of intermedite variables without backward: {}'.format(model._get_name(), total_nums))
    # print('Model {} : Number of intermedite variables with backward: {}'.format(model._get_name(), total_nums*2))
    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size*2 / 1000 / 1000))

if __name__ == '__main__':
    # model_name = "unext"
    model_name = "shformer_add"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("=> creating model %s" % model_name)
    model = build_network(model_name).to(device)
    # compute_speed(model, (1, 3, 352, 352), int(0), iteration=1000)
    modelsize(model, (1, 3, 512, 512))