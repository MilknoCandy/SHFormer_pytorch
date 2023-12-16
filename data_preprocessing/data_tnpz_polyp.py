"""
❤Descripttion: 
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-06 20:02:07
❤LastEditTime: 2023-12-16 20:47:15
❤Github: https://github.com/MilknoCandy
"""
import numpy as np
from PIL import Image
import cv2
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib.pyplot as plt
import albumentations as A

if __name__ == '__main__':
    img_pth = '/media/luo/new/SDW/all_dataset/polyp/TrainDataset/image'
    lab_pth = '/media/luo/new/SDW/all_dataset/polyp/TrainDataset/mask'

    target_base = '/media/luo/new/SDW/all_dataset/polyp'
    target_Tr = join(target_base, 'train')

    maybe_mkdir_p(target_Tr)

    """Image and segmentation of hybrid polyp data are PNG format"""
    # all images are resized to [352, 352] and savd as .npz
    H, W = 352, 352
    # size: [384,288] save as .npz
    training_cases = subfiles(lab_pth, suffix='.png', join=False)
    # save the original size for all images
    img_size = []
    
    for tr in training_cases:
        case_id = tr[:-4]

        # get path
        input_seg_file = join(lab_pth, tr)
        input_img_file = join(img_pth, case_id+'.png')
        assert all([isfile(input_seg_file), isfile(input_img_file)]), "%s" % case_id
        
        print(f"preprocessing case {case_id} ......")
        img = cv2.imread(input_img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg = cv2.imread(input_seg_file, flags=0)

        seg[seg<127] = 0
        seg[seg>127] = 255
        if len(np.unique(seg)) != 2:
            raise Exception("Wrong segmentation")

        img_size.append(seg.shape)
        img = A.resize(img, height=H, width=W, interpolation=cv2.INTER_LINEAR)
        seg = A.resize(seg, height=H, width=W, interpolation=cv2.INTER_NEAREST)[..., None]

        all_data = np.dstack((img, seg))

        # compress to npz
        np.savez_compressed(join(target_Tr, f"{case_id}.npz"), data=all_data)
        print("done.")
        
    save_pickle(img_size, join(target_Tr, "ori_size.pkl"))