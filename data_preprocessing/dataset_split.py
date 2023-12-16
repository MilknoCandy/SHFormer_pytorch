"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-28 15:29:32
❤LastEditTime: 2023-12-16 20:47:45
❤Github: https://github.com/MilknoCandy
"""
import numpy as np
import shutil
from batchgenerators.utilities.file_and_folder_operations import *

if __name__ == '__main__':
    img_pth = '/media/luo/new/SDW/all_dataset/ISIC2018/ISIC2018_Task1-2_Training_Input'
    lab_pth = '/media/luo/new/SDW/all_dataset/ISIC2018/ISIC2018_Task1_Training_GroundTruth'

    target_base = '/media/luo/new/SDW/all_dataset/ISIC2018/'
    target_imgsTr = join(target_base, "imagesTr")
    target_imgsTs = join(target_base, "imagesTs")
    target_labsTr = join(target_base, "labelsTr")
    target_labsTs = join(target_base, "labelsTs")

    maybe_mkdir_p(target_imgsTr)
    maybe_mkdir_p(target_imgsTs)
    maybe_mkdir_p(target_labsTr)
    maybe_mkdir_p(target_labsTs)
    
    cur = join(img_pth)
    all_files = subfiles(cur, suffix='.jpg', join=False)    # xxx.jpg in img_pth

    # data_split : 2000/594
    nums = len(all_files)

    ind = np.random.permutation(nums)

    # train set
    tr_inds = ind[:2000]
    tr_inds.sort()
    for i in tr_inds:
        p = all_files[i]
        tr_img = join(img_pth, p)
        tr_lab = join(lab_pth, p).replace('.jpg', '_segmentation.png')
        assert all([isfile(tr_img), isfile(tr_lab)]), "%s" % i

        print(f"moving file {p} ......")
        shutil.move(tr_img, target_imgsTr)
        shutil.move(tr_lab, target_labsTr)
        print(f"moving done.")

    # test set
    ts_inds = ind[2000:]
    ts_inds.sort()
    for i in ts_inds:
        p = all_files[i]
        ts_img = join(img_pth, p)
        ts_lab = join(lab_pth, p).replace('.jpg', '_segmentation.png')
        assert all([isfile(ts_img), isfile(ts_lab)]), "%s" % i

        print(f"moving file {p} ......")
        shutil.move(ts_img, target_imgsTs)
        shutil.move(ts_lab, target_labsTs)
        print(f"moving done.")