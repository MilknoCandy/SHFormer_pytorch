"""
❤Descripttion: 
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-06 20:02:07
❤LastEditTime: 2023-12-16 20:45:30
❤Github: https://github.com/MilknoCandy
"""
import numpy as np
from PIL import Image
import cv2
from batchgenerators.utilities.file_and_folder_operations import *
import matplotlib.pyplot as plt
import albumentations as A

if __name__ == '__main__':
    img_pth = '/root/autodl-tmp/CVC-ClinicDB_png/Original'
    lab_pth = '/root/autodl-tmp/CVC-ClinicDB_png/Ground Truth'

    target_base = '/root/autodl-tmp/CVC-ClinicDB_png'
    target_Tr = join(target_base, 'train')

    maybe_mkdir_p(target_Tr)

    """CVC-ClinicDB's image and segmentation are .tif"""
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
        seg = cv2.imread(input_seg_file, flags=0)[..., None]
        
        seg[seg<127] = 0
        seg[seg>127] = 255

        img_size.append(seg.shape)

        all_data = np.dstack((img, seg))

        # compress to npz
        np.savez_compressed(join(target_Tr, f"{case_id}.npz"), data=all_data)
        print("done.")
        
    save_pickle(img_size, join(target_Tr, "ori_size.pkl"))