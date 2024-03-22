"""
❤Descripttion: train or test SHFormer
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-30 09:13:08
❤LastEditTime: 2024-03-22 15:40:51
❤Github: https://github.com/MilknoCandy
"""
import os
import sys

now_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = now_dir.rsplit('/', 1)[0]
sys.path.append(root_dir)
import argparse

import yaml
from loguru import logger
# from memory_profiler import profile
from utils.default_configuration import get_default_configuration


def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=str, default='1')
    
    parser.add_argument("--config", type=str, default='configs/shformer_add.yaml', help='fill your own config location')
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true", default=False)
    parser.add_argument("--disable_saving", required=False, action='store_true',
                        help="If set, trainer will not save any parameter files. Useful for development when you are "
                            "only interested in the results and want to save some disk space")
    parser.add_argument("--pretrain_pth", required=False, type=str, default=False,
                        help="If set, trainer will load pretrained params")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    from mmcv import Config
    cfg_pth = args.config
    cfg = Config.fromfile(cfg_pth)

    outpath = cfg.output.outpath
    find_lr = False

    network = cfg.model.model_name
    network_trainer = cfg.training.trainer
    validation_only = cfg.training.validation_only

    deterministic = cfg.training.deterministic
    valbest = cfg.validation.valbest
    valpretrain = cfg.validation.valpretrain    # use pretrained params for testing

    run_mixed_precision = cfg.training.fp16

    fold = cfg.dataset.fold
    data_type = cfg.dataset.data_type
    assert data_type in ['0', '1', '2', '3'], "data_type must be 0: train/test, 1: train/val/test, 2: train, 3: single train/multiple test"
    
    dataset_directory = cfg.dataset.dataset_directory

    if fold == 'all':
        pass
    else:
        fold = int(fold)

    output_folder_name, trainer_class = get_default_configuration(
        outpath=outpath, network=network, network_trainer=network_trainer, dataset_directory=dataset_directory, fold=fold
    )

    # set trainer
    trainer = trainer_class(
        cfg=cfg, 
        fold=fold, 
        output_folder=output_folder_name, 
        dataset_directory=dataset_directory, 
        data_type=data_type, 
        deterministic = deterministic,
        sed=0,
        fp16=run_mixed_precision
    )

    if args.disable_saving:
        trainer.save_latest_only = False # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = False # whether or not to save checkpoint_latest
        trainer.save_best_checkpoint = False # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False # whether or not to save the final checkpoint

    # initialize trainer
    trainer.initialize(not validation_only)
    
    # load pretrianed params if you have one
    if args.pretrain_pth:
        trainer.load_pretrain_checkpoint(train=not validation_only, pretrained_pth=args.pretrain_pth)

    if not validation_only:
        if args.continue_training:
            trainer.load_latest_checkpoint()
        trainer.run_training()
    else:
        if valpretrain:
            pass
        elif valbest:
            trainer.load_best_checkpoint(train=False)
        else:
            trainer.load_final_checkpoint(train=False)

    trainer.validate()
    # trainer.output_seg(save_output_folder="predict_results")
    # trainer.eval_speed(input_size=(1, 3, 512, 512))
    # trainer.uncertain_visualize(output_file="uncertainty_visualize")
    # trainer.featmap_vis(output_file="featmap_visualize", method="gradcam", layer_aggregate=True)
    # trainer.featmap_vis(output_file="mask_visualize")

if __name__ == '__main__':
    main()