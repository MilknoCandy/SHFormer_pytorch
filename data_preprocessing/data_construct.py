import shutil

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image

if __name__ == "__main__":
    img_pth = '/media/luo/new/SDW/all_dataset/ISIC2018/ISIC2018_Task1-2_Training_Input'
    lab_pth = '/media/luo/new/SDW/all_dataset/ISIC2018/ISIC2018_Task1_Training_GroundTruth'

    target_base = '/media/luo/new/SDW/all_dataset/ISIC2018'
    target_database = join(target_base, 'train')
    maybe_mkdir_p(target_database)

    img_files = subfiles(img_pth, suffix=".jpg", join=False)
    lab_files = subfiles(lab_pth, suffix=".png", join=False)

    for img, lab in zip(img_files, lab_files):
        case_id = img[:-4]
        print(f"Compressing {case_id}......")
        if isfile(join(target_database, f"{case_id}.npz")):
            os.unlink(join(img_pth, img))
            os.unlink(join(lab_pth, lab))
            print(f"data {case_id} is already exists.")
            continue
        image = np.array(Image.open(join(img_pth, img)))
        label = np.array(Image.open(join(lab_pth, lab)))[..., None]
        data = np.dstack((image, label))
        os.unlink(join(img_pth, img))
        os.unlink(join(lab_pth, lab))
        # compress to npz file
        np.savez_compressed(join(target_database, f"{case_id}.npz"), data=data)
