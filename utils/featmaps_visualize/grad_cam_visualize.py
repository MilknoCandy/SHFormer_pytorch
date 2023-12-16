import random

import cv2
import torch
from alive_progress import alive_bar
from batchgenerators.utilities.file_and_folder_operations import *
from pytorch_grad_cam import (AblationCAM, EigenCAM, EigenGradCAM, FullGrad,
                              GradCAM, GradCAMPlusPlus, LayerCAM, ScoreCAM,
                              XGradCAM)
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image
from torch import nn
from torch.utils.data import Dataset
from utils.dict_tools.dict_combine import combine_deepest_dict
from network_architecture.simImple import resize
from pytorch_grad_cam.ablation_layer import AblationLayerVit


class SemanticSegmentationTarget:
    def __init__(self, category, mask) -> None:
        self.category = category
        self.mask = mask
    
    def __call__(self, model_output):
        """
        Args:
            model_output: num_classes x H x W
        """
        # 由于测试时输入分辨率为[512, 512], 而mask分辨率为原分辨率, 因此将输出resize至mask分辨率
        # model_output = resize(model_output.unsqueeze(0), size=self.mask.size(), mode='bilinear').squeeze()
        return (model_output[self.category, :, :] * self.mask).sum()

class ReshapeTransform:
    def __init__(self, down_scale={32:4, 64: 8, 160: 16, 256: 32}, img_size=(512, 512)) -> None:
        self.down_scale = down_scale
        self.img_size = img_size

    def reshape_transform(self, tensor_, height=512, width=512):
        '''
        GradCam是对2D图像进行可视化, 因此需要输出为具体需要查看所对应的配置文件yaml
        height = width = config.DATA.IMG_SIZE / config.MODEL.NUM_HEADS[-1]
        比如该例子中IMG_SIZE: 224  NUM_HEADS: [4, 8, 16, 32]
        height = width = 224 / 32 = 7
        '''
        if len(tensor_.shape) != 4:
            # 对transformer的特征需要转换
            result = tensor_.reshape(tensor_.size(0),
                                    height, width, tensor_.size(2))

            # Bring the channels to the first dimension,
            # like in CNNs.
            # result = result.transpose(2, 3).transpose(1, 2)
            result = result.permute(0, 3, 1, 2)
            return result
        else:
            return tensor_

    
    def __call__(self, tensor):
        if isinstance(tensor, tuple):
            tensor = tensor[-1]

        stage_scale = self.down_scale[tensor.size(-1)]  if len(tensor.shape) != 4 else self.down_scale[tensor.size(1)]
        H, W = self.img_size[0]//stage_scale, self.img_size[1]//stage_scale

        return self.reshape_transform(tensor, height=H, width=W)

def hotmap_gradcam(model, method, module, data, down_scale, img_size):
    """绘制指定层输出的特征图的根据信息熵计算得到的uncertainty的特征图

    Args:
        model (nn.Module): 构建好的网络(已加载训练好的参数)
        method : gradcam类, 用以计算梯度图(这里仅考虑GradCam)
        module : 指定的网络层, 获取该层输出特征图
        data (tuple): 包含input_tensor(网络输入图像, 已处理好并转为Tensor类型)、mask(地面真值分割图)和imgo(原图像, 已归一化且转为numpy格式)
    """
    input_tensor, mask, imgo = data
    # 使用cam
    # NOTE: 如果一次给多层, 则是多层的梯度聚合至一层
    # 仅考虑gradcam
    cams = method(
        model=model,
        target_layers=module,
        use_cuda=torch.cuda.is_available(),
        reshape_transform=ReshapeTransform(down_scale=down_scale, img_size=img_size)
    )
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    targets = [SemanticSegmentationTarget(1, mask.squeeze())]
    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cams.batch_size = 1
    with cams as cam: 
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets ,
                            # Reduce noise by taking the first principle componenet of cam_weights*activations
                            eigen_smooth=False, 
                            # Apply test time augmentation to smooth the CAM
                            aug_smooth=False)
    
        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_img = show_cam_on_image(imgo, grayscale_cam, use_rgb=False)
    return cam_img

def gradcam_visualize(model, output_file: str, data_generator: Dataset, img_size: tuple=(512, 512), method: str="gradcam", layer_aggregate=False):
    """使用GradCam绘制指定层输出的特征图的根据信息熵计算得到的uncertainty的特征图

    Args:
        model (_type_): _description_
        output_file (str): _description_
        data_generator (Dataset): _description_
        img_size (tuple, optional): _description_. Defaults to (512, 512).
    """
    # gradcam提供的方法
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }
    if method not in methods.keys():
        raise Exception(f"methos should be one of {methods.keys()}")

    # 可视化网络的特征图
    # 创建特征图可视化存放文件夹
    maybe_mkdir_p(output_file)
    model.eval()

    # 由于transformer的输出都是序列, 需要重新reshape, 而从头到尾分别下采样了4/8/16/32倍
    down_scale = {32:4, 64: 8, 160: 16, 256: 32}

    # 需要可视化的实例索引
    data_nums = len(data_generator)
    # ind_select = random.sample(range(data_nums), 10)    # [List]
    # 7, 13, 47, 120, 121
    ind_select = [7, 47, 51, 79, 121, 289, 294, 300, 304, 368, 488]
    ind_select = set(ind_select)                        # [Set]

    # 选择需要可视化的层
    # 最深层的值为该层输出的特征图的名字
    layers_select_with_name = {
            "backbone": {
                # "norm1": "sa1",
                # "norm2": "sa2",
                # "norm3": "sa3",
                # "norm4": "sa4",
                # "linear1": "channel1", "conv1": "spatial1", "fe": "sc1"， "mlp": "fc2"
                "block1": {"0": {"mlp": {"fe": {"linear1": "channel1"}}}},
                "block2": {"0": {"mlp": {"fe": {"linear1": "channel2"}}}},
                "block3": {"0": {"mlp": {"fe": {"linear1": "channel3"}}}},
                "block4": {"0": {"mlp": {"fe": {"linear1": "channel4"}}}},
            }
                }
    layers = combine_deepest_dict(layers_select_with_name, [])
    # 循环获取网络层的hook
    feat_names = list()
    layers_module = list()
    # 获取网络层不能在案例循环中, 这样会每次多获取一次, 所以receptive那个函数里先获取网络层再在循环中hook
    for layer_and_ftname in layers:
        # 将层的名字和特征图的名字分开
        layer, ftname = layer_and_ftname.rsplit('.', 1)
        feat_names.append(ftname)   # 添加到列表供绘图使用
        # 对指定层获取特征图(如果为Sequential则会直接获得最终输出图, 
        # 若为ModuleList则需要获取指定层, 当然通常为最后一层)
        layers_module.append(model.get_submodule(layer))

    # 遍历测试集图像
    print("="*20, f"output Testing hotmap_featmaps", "="*20)
    with alive_bar(data_nums, spinner='twirl', length=80, force_tty=True, title="Output Testing featmaps") as bar:  
        for i, data in enumerate(data_generator):

            if i not in ind_select:
                bar()
                continue
            
            input_tensor, input_gt, _, imgo = data
            # 由于mask用的是原分辨率, 因此使用Nearest进行resize
            input_gt = resize(input_gt, size=img_size, mode='nearest')

            if torch.cuda.is_available():
                model = model.cuda()
                input_tensor = input_tensor.cuda()
                input_gt = input_gt.cuda()
            imgo = resize(imgo.permute(0, 3, 1, 2) / 255, size=img_size, mode='bilinear').permute(0, 2, 3, 1)
            imgo = imgo.squeeze().numpy() 
                
            if layer_aggregate:
                cam_image = hotmap_gradcam(
                    model=model, method=methods[method], module=layers_module,
                    data=(input_tensor, input_gt, imgo),
                    down_scale=down_scale, img_size=img_size
                )
                cv2.imwrite(join(output_file, f"{i+1}_featmap_all_{method}_cam.jpg"), cam_image)
            else:
                for model_module, feat_name in zip(layers_module, feat_names):
                    cam_image = hotmap_gradcam(
                        model=model, method=methods[method], module=[model_module],
                        data=(input_tensor, input_gt, imgo),
                        down_scale=down_scale, img_size=img_size
                    )
                    cv2.imwrite(join(output_file, f"{i+1}_featmap_{feat_name}_{method}_cam.jpg"), cam_image)
                    
            bar()
