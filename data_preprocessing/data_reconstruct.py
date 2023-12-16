import shutil
from batchgenerators.utilities.file_and_folder_operations import *


if __name__ == '__main__':
    target_base = '/media/luo/new/SDW/all_dataset/ISIC2018/'
    target_imgsTr = join(target_base, "imagesTr")
    target_imgsTs = join(target_base, "imagesTs")
    target_labsTr = join(target_base, "labelsTr")
    target_labsTs = join(target_base, "labelsTs")

    img_pth = '/media/luo/new/SDW/all_dataset/ISIC2018/ISIC2018_Task1-2_Training_Input'
    lab_pth = '/media/luo/new/SDW/all_dataset/ISIC2018/ISIC2018_Task1_Training_GroundTruth'

    train_img_files = subfiles(target_imgsTr, suffix=".jpg", join=False)

    for f in train_img_files:
        print(f"moving {f}......")
        f = join(target_imgsTr, f)
        assert isfile(f), "wrong"
        shutil.move(f, img_pth)
        print("done.")
    
    test_img_files = subfiles(target_imgsTs, suffix=".jpg", join=False)

    for f in test_img_files:
        print(f"moving {f}......")
        f = join(target_imgsTs, f)
        assert isfile(f), "wrong"
        shutil.move(f, img_pth)
        print("done.")

    train_lab_files = subfiles(target_labsTr, suffix=".png", join=False)

    for f in train_lab_files:
        print(f"moving {f}......")
        f = join(target_labsTr, f)
        assert isfile(f), "wrong"
        shutil.move(f, lab_pth)
        print("done.")
    
    test_lab_files = subfiles(target_labsTs, suffix=".png", join=False)

    for f in test_lab_files:
        print(f"moving {f}......")
        f = join(target_labsTs, f)
        assert isfile(f), "wrong"
        shutil.move(f, lab_pth)
        print("done.")
