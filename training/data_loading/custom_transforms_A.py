"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-01 20:45:11
❤LastEditTime: 2023-12-16 21:14:15
❤Github: https://github.com/MilknoCandy
"""
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import DualTransform, ImageOnlyTransform
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from training.data_loading.custom_transform_interface import MaskOnlyTransform

__all__ = [
    "Custom_resize",
    "Test_resize",
    "Custom_random_scale_crop",
    "Custom_random_vertical_flip",
    "Custom_random_horizontal_flip",
    "Custom_random_rotate",
    "Custom_random_image_enhance",
    "Custom_random_dilation_erosion",
    "Custom_random_gaussian_blur",
]
class Custom_resize(DualTransform):
    def __init__(self, size_h, size_w, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.size_h = size_h
        self.size_w = size_w

    def apply(self, img, **params):
        img = A.resize(img, width=self.size_w, height=self.size_h)    # default is bilinear

        return img
    
    def get_transform_init_args(self):
        return {
            "size_h": self.size_h,
            "size_w": self.size_w
        }

class Test_resize(ImageOnlyTransform):
    def __init__(self, size_h, size_w, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.size_h = size_h
        self.size_w = size_w

    def apply(self, img, **params):
        img = A.resize(img, width=self.size_w, height=self.size_h)    # default is bilinear

        return img
    
    def get_transform_init_args(self):
        return {
            "size_h": self.size_h,
            "size_w": self.size_w
        }

class Custom_random_scale_crop(DualTransform):
    def __init__(self, range=[0.75, 1.25], always_apply: bool = False, p: float = 0.5):
        """_summary_

        Args:
            range (list, optional): _description_. Defaults to [0.75, 1.25].
            always_apply (bool, optional): _description_. Defaults to False.
            p (float): probability of applying the transform. Default: 0.5.
        """
        super().__init__(always_apply, p)
        self.range = range
        # self.scale = np.random.random() * (self.range[1] - self.range[0]) + self.range[0]

    def adjust(self, img, box):
        x0, y0, x1, y1 = box

        W1 = max(x1, img.shape[1])
        W0 = min(x0, 0)
        H1 = max(y1, img.shape[0])
        H0 = min(y0, 0)

        if len(img.shape)==3:
            img_ = np.zeros((H1-H0, W1-W0, img.shape[2]), dtype=img.dtype)
        else:
            img_ = np.zeros((H1-H0, W1-W0), dtype=img.dtype)

        W0 = abs(W0)
        H0 = abs(H0)
        img_[H0:H0+img.shape[0], W0:W0+img.shape[1]] = img
        return img_

    def apply(self, img, scale=0, **params):
        # base_size = sample[key].size    # (width, height)
        base_size = img.shape[0], img.shape[1]

        scale_size = tuple((np.array(base_size) * scale).round().astype(int))
        # sample[key] = sample[key].resize(scale_size)
        img = A.resize(img, height=scale_size[0], width=scale_size[1])

        x0 = (img.shape[1] - base_size[1]) // 2
        y0 = (img.shape[0] - base_size[0]) // 2
        x1 = (img.shape[1] + base_size[1]) // 2
        y1 = (img.shape[0] + base_size[0]) // 2
        box = x0, y0, x1, y1
        img = self.adjust(img, box)

        if scale > 1:
            x0 = abs(x0)
            y0 = abs(y0)
            x1 = abs(x1)
            y1 = abs(y1)

        else:
            x0 = 0
            y0 = 0
            x1 = img.shape[1]
            y1 = img.shape[0]

        img = A.crop(
            img=img,
            x_min=x0,
            y_min=y0,
            x_max=x1,
            y_max=y1
        )

        return img

    def get_params(self):
        return {
            "scale": np.random.random() * (self.range[1] - self.range[0]) + self.range[0],
        }

    def get_transform_init_args(self):
        return {
            "range": self.range
        }


class Custom_random_vertical_flip(DualTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        
    def apply(self, img, **params):
        img = np.fliplr(img)
        return img

    
class Custom_random_horizontal_flip(DualTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
    
    def apply(self, img, **params):
        img = np.flipud(img)
        return img


class Custom_random_rotate(DualTransform):
    def __init__(self, range=[0, 360], interval=1, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.range = range
        self.interval = interval

    def apply(self, img, rot=0, **params):
        rot = rot + 360 if rot < 0 else rot
        base_size = img.shape[0], img.shape[1]

        img = A.rotate(img=img, angle=rot, interpolation=cv2.INTER_NEAREST)

        img = A.crop(
            img=img,
            x_min=(img.shape[1] - base_size[1]) // 2,
            y_min=(img.shape[0] - base_size[0]) // 2,
            x_max=(img.shape[1] + base_size[1]) // 2,
            y_max=(img.shape[0] + base_size[0]) // 2
        )                                                                  

        return img

    def get_params(self):
        return {
            "rot": (np.random.randint(*self.range) // self.interval) * self.interval,
        }

    def get_transform_init_args(self):
        return {
            "range": self.range,
            "interval": self.interval
        }

class Custom_random_image_enhance(ImageOnlyTransform):
    def __init__(self, methods=['contrast', 'brightness', 'sharpness'], always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.enhance_method = []
        if 'contrast' in methods:
            self.enhance_method.append(ImageEnhance.Contrast)
        if 'brightness' in methods:
            self.enhance_method.append(ImageEnhance.Brightness)
        if 'sharpness' in methods:
            self.enhance_method.append(ImageEnhance.Sharpness)

    def apply(self, img, **params):
        # 未在albumentations中找到替代方法
        # img为numpy, 先转为PIL, 处理后转回numpy
        img = Image.fromarray(img)
        np.random.shuffle(self.enhance_method)      # 打乱顺序

        for method in self.enhance_method:
            if np.random.random() > 0.5:
                enhancer = method(img)
                factor = float(1 + np.random.random() / 10)
                img = enhancer.enhance(factor)
        img = np.array(img)

        return img

class Custom_random_dilation_erosion(MaskOnlyTransform):
    def __init__(self, kernel_range, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply, p)
        self.kernel_range = kernel_range

    def apply_to_mask(self, gt, **params):
        # gt = sample['gt']
        # gt = np.array(gt)
        key = np.random.random()
        # kernel = np.ones(tuple([np.random.randint(*self.kernel_range)]) * 2, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (np.random.randint(*self.kernel_range), ) * 2)
        if key < 1/3:
            gt = cv2.dilate(gt, kernel)
        elif 1/3 <= key < 2/3:
            gt = cv2.erode(gt, kernel)

        # sample['gt'] = Image.fromarray(gt)

        return gt

    def get_transform_init_args(self):
        return {
            "kernel_range": self.kernel_range
        }

class Custom_random_gaussian_blur(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        # 输入为numpy, 先转PIL, 处理后转回numpy
        img = Image.fromarray(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=np.random.random()))
        img = np.array(img)

        return img
