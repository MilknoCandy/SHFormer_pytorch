"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-30 19:46:43
❤LastEditTime: 2023-03-06 11:32:30
❤Github: https://github.com/MilknoCandy
"""
import os
import sys

from batchgenerators.utilities.file_and_folder_operations import *
from utils.model_restore import recursive_find_python_class
from loguru import logger

def get_default_configuration(outpath, network, network_trainer, dataset_directory, fold, \
        search_in=(sys.path[-1], "training", "trainer"), \
        module='training.trainer'):
    """all training setting representation

    Args:
        outpath (_type_): _description_
        network (_type_): _description_
        network_trainer (_type_): _description_
        dataset_directory (_type_): _description_
        fold (_type_): _description_

    Returns:
        _type_: _description_
    """

    task = dataset_directory.rsplit('/', 2)[-1]
    network_training_output_dir = join(sys.path[-1], "results")

    fold = f"fold_{fold}"
    output_folder_name = join(network_training_output_dir, network, task, fold, outpath)
    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer, current_module=module)

    # 添加log文件
    logger.add(join(output_folder_name, "trainer_log.log"))
    # print("#" * 100)
    # print(f"I am running the following Net: {network}")
    logger.info(
        "\n#############################################################################"
        f"\nI am running the following Net: {network}"
        f"\nMy trainer class is: {trainer_class}"
        f"\nI am using data from this folder: {dataset_directory}\n"
        "#############################################################################"
    )
    # print("My trainer class is: ", trainer_class)
    # print("\nI am using data from this folder: ", dataset_directory)
    # print("#" * 100)

    return output_folder_name, trainer_class