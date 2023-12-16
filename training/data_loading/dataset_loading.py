"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-30 20:44:27
❤LastEditTime: 2023-12-16 21:17:16
❤Github: https://github.com/MilknoCandy
"""
import os
from collections import OrderedDict

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from batchgenerators.utilities.file_and_folder_operations import join
from PIL import Image
from torch.utils.data import Dataset
from training.data_loading.custom_transforms_A import *
from utils.mask2onehot import mask2onehot


def get_case_identifiers(folder, data_suffix):
    """only for getting the unique IDs for each data

    Args:
        folder (str): _description_

    Returns:
        _type_: _description_
    """
    case_identifiers = [i[:-4] for i in os.listdir(folder) if i.endswith(data_suffix)]
    return case_identifiers

def load_dataset(folder, data_type, data_suffix):
    """the training process is using xx.npz and the testing process is using .npz, jpg, png"""
    # we don't load the actual data but instead return the filename to the np file.
    if data_type == '0':
        data_mode = ['train', 'test']
    elif data_type == '1':
        data_mode = ['train', 'val', 'test']
    elif data_type == '2':
        data_mode = ['train']
    elif data_type == '3':
        data_mode = ['train', 'test_mul']

    # images_fodlers = {
    #     "train": "imagesTr",
    #     "val": "imagesVal",
    #     "test": "imagesTs",
    # }
    # labels_fodlers = {
    #     "train": "labelsTr",
    #     "val": "labelsVal",
    #     "test": "labelsTs",
    # }
    print('loading dataset...')
    case_identifiers = OrderedDict()
    for m in data_mode:
        """dataset structure
        imagesTr/labelsTr
        imagesVal/labelsVal
        imagesTs/labelsTs
        """
        if m == 'test_mul':
            continue
        case_identifiers[m] = get_case_identifiers(join(folder, m), data_suffix)
        case_identifiers[m].sort()

    dataset = OrderedDict()
    if data_type=="2":
        """if data_type is train, we need to separate it to train/test
        """
        for m in data_mode:
            for c in case_identifiers[m]:
                dataset[c] = OrderedDict()
                dataset[c]['data_file'] = join(folder, m, f"{c}.npz")

    else:
        for m in data_mode:
            if m == 'test_mul':
                continue
            dataset[m] = OrderedDict()
            for c in case_identifiers[m]:
                dataset[m][c] = OrderedDict()
                dataset[m][c]['data_file'] = join(folder, m, f"{c}.npz")
    if data_type == '3':
        return dataset, True
    else:
        return dataset

class CustomDataSet(Dataset):
    def __init__(self, dataset, transform_list, oneHot=False):

        self._data = dataset
        self.case_ids = list(dataset.keys())
        self.oneHot = oneHot
        if transform_list.NO_custom:
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
                A.ColorJitter(),
                A.HorizontalFlip(),
                A.VerticalFlip(),      
            ])
        else:
            self.transform = self.get_transform(transform_list)

        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.gt_transform = T.Compose([
            T.ToTensor()
        ])
        

    def __getitem__(self, index):
        all_data = np.load(self._data[self.case_ids[index]]['data_file'])['data']
        img = all_data[..., :-1]
        lab = all_data[..., -1]
        
        transformed = self.transform(image=img, mask=lab)

        img = self.img_transform(transformed['image'])
        lab = self.gt_transform(transformed['mask'])

        if self.oneHot:
            classes = torch.unique(lab)
            num_classes = len(classes)
            lab = mask2onehot(lab, num_classes)
        return img, lab

    def __len__(self):
        return len(self._data)

        
    @staticmethod
    def get_transform(transform_list):
        tfs = []
        # tfs = A.Compose([])
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if key=="NO_custom":
                continue
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return A.Compose(tfs)

class TestDataSet(Dataset):
    def __init__(self, dataset, img_size, crop_size, transform_list, oneHot=False):
        self._data = dataset
        self.case_ids = list(dataset.keys())
        self.oneHot = oneHot
        self.transform = self.get_transform(transform_list)
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.gt_transform = T.Compose([
            T.ToTensor()
        ])
        self.image_size = img_size
        

    def __getitem__(self, index):

        name = self.case_ids[index]
        imgo = cv2.imread(self._data[name][0])
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self._data[name][1], flags=0)
                 
        transformed = self.transform(image=imgo, mask=label)
        img = self.img_transform(transformed['image'])
        label = self.gt_transform(transformed['mask'])

        if self.oneHot:
            classes = torch.unique(label)
            num_classes = len(classes)
            label = mask2onehot(label, num_classes)

        return img, label, self.image_size[index], imgo

    def __len__(self):
        total_img = len(self._data)
        return total_img
        
    @staticmethod
    def get_transform(transform_list):
        tfs = []
        # tfs = A.Compose([])
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if key=="NO_custom":
                continue
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return A.Compose(tfs)

class MulTestDataset(Dataset):
    def __init__(self, img_pth, lab_pth, transform_list, one_Hot=False) -> None:
        super().__init__()
        self.img_pth = img_pth
        self.lab_pth = lab_pth
        self.oneHot = one_Hot
        self.img_files = self.read_file(self.img_pth)
        self.lab_files = self.read_file(self.lab_pth)
        self.img_size = []

        self.transform = self.get_transform(transform_list)
        self.img_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.gt_transform = T.Compose([
            T.ToTensor()
        ])
        self._init_img_size()

    def __getitem__(self, index):
        imgo = cv2.imread(self.img_files[index])
        imgo = cv2.cvtColor(imgo, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.lab_files[index], flags=0)

        transformed = self.transform(image=imgo, mask=label)
        img = self.img_transform(transformed['image'])
        label = self.gt_transform(transformed['mask'])

        if self.oneHot:
            classes = torch.unique(label)
            num_classes = len(classes)
            label = mask2onehot(label, num_classes)

        return img, label, self.img_size[index], imgo
    
    def __len__(self):
        total_img = len(self.img_files)
        return total_img
    
    def _init_img_size(self):
        for i in range(self.__len__()):
            img = cv2.imread(self.img_files[i])
            self.img_size.append(img.shape[:2])
    
    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        file_path_list.sort()
        return file_path_list

    @staticmethod
    def get_transform(transform_list):
        tfs = []
        # tfs = A.Compose([])
        for key, value in zip(transform_list.keys(), transform_list.values()):
            if key=="NO_custom":
                continue
            if value is not None:
                tf = eval(key)(**value)
            else:
                tf = eval(key)()
            tfs.append(tf)
        return A.Compose(tfs)