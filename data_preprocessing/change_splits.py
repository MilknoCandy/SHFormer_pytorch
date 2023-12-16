"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-12-08 20:40:41
❤LastEditTime: 2022-12-08 21:49:22
❤Github: https://github.com/MilknoCandy
"""
from batchgenerators.utilities.file_and_folder_operations import *
best_splits = load_pickle("/media/luo/new/SDW/all_dataset/data_preprocessed/Task004_ISIC2018/splits_all.pkl")
now_splist = load_pickle("/media/luo/new/SDW/all_dataset/ISIC2018/splits_all.pkl")
now_splist[-1]['train'] = best_splits[-1]['train']
now_splist[-1]['val'] = best_splits[-1]['val']
now_splist[-1]['test'] = best_splits[-1]['test']
save_pickle(now_splist, "/media/luo/new/SDW/all_dataset/ISIC2018/splits_all.pkl")
print("test")