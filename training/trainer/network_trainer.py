"""
❤Descripttion: adapted from here: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/network_training/network_trainer.py
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-29 10:51:56
❤LastEditTime: 2023-11-24 09:20:12
❤Github: https://github.com/MilknoCandy
"""
import sys
import time
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from warnings import warn

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm, trange
from utils.to_torch import maybe_to_torch, to_cuda

matplotlib.use('agg')

class NetworkTrainer(object):
    def __init__(self, data_type='0', output_folder=None, dataset_directory=None, sed=1, deterministic=True, fp16=False):
        """
        A generic class that provides basic functionality. Training can be terminated early if the validation loss (or the target metric
        if implemented) do not improve anymore. This is based on a moving average (MA) of the loss/metric instead of the raw values to get more smooth
        results.

        This code includes:
        - 
        What you need to override:
        - __init__
        - initialize
        - run_online_evaluation (optional)
        - finish_online_evaluation (optional)
        - validate
        - predict_test_case
        """
        # 混合精度训练加速
        self.fp16 = fp16
        self.amp_grad_scaler = None

        if deterministic:
            # this is too long ~about 6 hours for only 1.94m model!!!
            random.seed(sed)    # NOTE: python的random
            np.random.seed(sed)
            torch.manual_seed(sed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sed)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        ###################### SET THESE IN self.initialize() #########################################
        # initialize trainer's parameters
        self.lr_scheduler = None
        self.network = None         # nnUNet设置的有多GPU模式，暂不考虑
        self.optimizer = None
        self.tr_gen = self.val_gen = None   # set generator
        self.was_initialized = False
        ############################ SET THESE IN INIT ################################################
        self.dataset_directory = dataset_directory
        self.fold = None
        self.loss = None
        self.output_folder = output_folder
        ######################## SET THESE IN LOAD_DATASET OR DO_SPLIT ################################
        self.dataset = None  # these can be None for inference mode

        # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split
        self.dataset_tr = self.dataset_val = None  
        # for test
        self.dataset_ts = None
        self.ts_mul = False
        ################################# SETTING FOR TRAINING ########################################
        self.also_val_in_tr_mode = False
        self.max_num_epochs = 100
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

        self.patience = 50  # NOTE: default is 50
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4
        self.val_eval_criterion_alpha = 0.9
        #################################### LOSS params ##############################################
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []      # does not have to be used
        # NOTE: nnunet根据训练集的loss来确定是否early stop, 加多一个根据验证集的指标进行判断
        self.best_epoch_based_MA_tr_loss = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_MA_val_eval_criterion = None
        self.best_MA_val_eval_criterion_for_patience = None
        self.best_val_eval_criterion_MA = None

        self.epoch = 0
        self.log_file = None
        self.train_loss_MA = None
        self.use_progress_bar = True
        self.val_eval_criterion_MA = None
        ########################### Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint
        ################################# CUSTOMIZED ###################################################
        self.data_type = data_type
        self.all_size = None        # 存放数据所有size(仅当data_type='2'时起作用)
        self.ts_use_npz = False      # 为True则使用resize后的npz文件进行测试; False则使用原图像进行测试
        if not self.ts_use_npz:
            self.val_size = None
            self.ts_size = None
            self.ts_img_pth = None
            self.ts_lab_pth = None

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []    

    @abstractmethod
    def initialize(self, training=True):
        """
        create self.output_folder

        modify self.output_folder if you are doing cross-validation (one folder per fold)

        set self.tr_gen and self.val_gen

        call self.initialize_network and self.initialize_optimizer_and_scheduler (important!)

        finally set self.was_initialized to True
        """
    
    @abstractmethod
    def load_dataset(self):
        """
        This step generates a 'list' of the files in the dataset, which is actually a 'dict' of the filename 
        of each image in the dataset and the corresponding filename of the description file and the contents 
        of the description file. The images themselves or the array of images are still not read in.
        """
        pass

    @abstractmethod
    def initialize_network(self):
        """initialize self.network here.
        """
        pass

    @abstractmethod
    def initialize_optimizer_and_scheduler(self):
        """initialize self.optimizer and self.lr_scheduler (if applicable) here.
        """
        pass

    def do_split(self, dsplit='tt'):
        """
        After reading all the images and their descriptions within the dataset, at this point there is no division 
        between the training and validation samples. The preprocessing stage generates a file called 'splits_final.pkl', 
        which contains the training-validation division for each fold of the cross-validation, so in this step the 
        list is split into training and validation parts according to the division table generated earlier.
        Condition: your dataset is a dictionary.
        """
        if self.data_type == '0':
            """data_type is train/test
            """
            # no fold
            tr_keys = list(self.dataset['train'].keys())
            # for test
            ts_keys = list(self.dataset['test'].keys())

            tr_keys.sort()
            ts_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset['train'][i]

            self.dataset_ts = OrderedDict()
            for i in ts_keys:
                self.dataset_ts[i] = self.dataset['test'][i]
            
            # since we don't have val, so we set val=test
            self.dataset_val = deepcopy(self.dataset_ts)

        elif self.data_type == '1':
            """train/val/test
            """
            # no fold
            tr_keys = list(self.dataset['train'].keys())
            val_keys = list(self.dataset['val'].keys())
            # for test
            ts_keys = list(self.dataset['test'].keys())

            tr_keys.sort()
            val_keys.sort()
            ts_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset['train'][i]

            self.dataset_val = OrderedDict()
            for i in val_keys:
                self.dataset_val[i] = self.dataset['val'][i]

            self.dataset_ts = OrderedDict()
            for i in ts_keys:
                self.dataset_ts[i] = self.dataset['test'][i]
        
        elif self.data_type == '2':
            """data_type is train, Create Folds
            """
            splits_file = join(self.dataset_directory, "splits_folds.pkl")
            img_size = load_pickle(self.all_size)
            if not isfile(splits_file):
                self.print_to_log_file("Creating new split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=41)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]

                    test_sizes = np.array(img_size)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['test'] = test_keys
                    splits[-1]['test_size'] = test_sizes.tolist()
                    
                save_pickle(splits, splits_file)
            
            splits = load_pickle(splits_file)

            if self.fold == "all":
                # 仅有train, 所以自行划分
                # 保存下来供测试使用
                splits_file = join(self.dataset_directory, "splits_all.pkl")
                if not isfile(splits_file):
                    splits_all = []
                    splits_all.append(OrderedDict())
                    # tr_keys = val_keys = list(self.dataset.keys())
                    ind = torch.randperm(len(self.dataset.keys()))
                    data_sum = len(ind)

                    # dsplit: 'tt' for train/test(default), 'tvt' for train/val/test
                    if dsplit=='tt':
                        """2000/594 is only for ISIC2018"""
                        tr_keys =  [list(self.dataset.keys())[i] for i in ind[:2000]]
                        val_keys =  [list(self.dataset.keys())[i] for i in ind[2000:]]

                        """
                        如果使用原图像进行测试, 那么需要原图像的size用于上采样
                        """
                        val_size = [img_size[i] for i in ind[2000:]]
                        splits_all[-1]['val_size'] = val_size
                        splits_all[-1]['test_size'] = val_size

                        ts_keys = val_keys

                    elif dsplit=='tvt':
                        tr_length = round(data_sum*0.8)
                        val_length = round(data_sum*0.1)
                        ts_length = data_sum - val_length - tr_length
                        tr_keys =  [list(self.dataset.keys())[i] for i in ind[:tr_length]]
                        val_keys =  [list(self.dataset.keys())[i] for i in ind[tr_length:tr_length+val_length]]
                        ts_keys =  [list(self.dataset.keys())[i] for i in ind[-ts_length:]]

                        """
                        如果使用原图像进行测试, 那么需要原图像的size用于上采样
                        """
                        val_size = [img_size[i] for i in ind[tr_length:tr_length+val_length]]
                        ts_size = [img_size[i] for i in ind[-ts_length:]]
                        splits_all[-1]['val_size'] = val_size
                        splits_all[-1]['test_size'] = ts_size

                    splits_all[-1]['train'] = tr_keys
                    splits_all[-1]['val'] = val_keys
                    splits_all[-1]['test'] = ts_keys
                    save_pickle(splits_all, splits_file)
                
                splits_all = load_pickle(splits_file)
                tr_keys = splits_all[-1]['train']
                val_keys = splits_all[-1]['val']
                ts_keys = splits_all[-1]['test']
                if not self.ts_use_npz:
                    val_size = splits_all[-1]['val_size']
                    ts_size = splits_all[-1]['test_size']

            else:
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['test']
                ts_keys = splits[self.fold]['test']
                if not self.ts_use_npz:
                    val_size = splits[self.fold]['test_size']
                    ts_size = splits[self.fold]['test_size']
                # val_keys = splits[self.fold]['val']
                # ts_keys = splits[self.fold]['val']

            tr_keys.sort()
            if not self.ts_use_npz:
                """
                如果使用原图像进行测试, 则在sort的时候需要将size和keys同步sort
                """
                val_keys, self.val_size = zip(*sorted(zip(val_keys, val_size)))
                ts_keys, self.ts_size = zip(*sorted(zip(ts_keys, ts_size)))

            else:
                val_keys.sort()
                ts_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset[i]

            self.dataset_val = OrderedDict()
            self.dataset_ts = OrderedDict()

            if not self.ts_use_npz:
                for i in val_keys:
                    self.dataset_val[i] = join(self.ts_img_pth, f"{i}.jpg"), join(self.ts_lab_pth, f"{i}_segmentation.png")
                
                if dsplit=='tt':
                    self.dataset_ts = deepcopy(self.dataset_val)
                
                elif dsplit=='tvt':
                    for i in ts_keys:
                        self.dataset_ts[i] = join(self.ts_img_pth, f"{i}.jpg"), join(self.ts_lab_pth, f"{i}_segmentation.png")

            else:
                for i in val_keys:
                    self.dataset_val[i] = self.dataset[i]

                if dsplit=='tt':
                    self.dataset_ts = deepcopy(self.dataset_val)
                elif dsplit=='tvt':
                    for i in ts_keys:
                        self.dataset_ts[i] = self.dataset[i]

        elif self.data_type == '3':
            """说明拥有多个测试集, 单个训练集"""
            # train set
            tr_keys = list(self.dataset['train'].keys())
            tr_keys.sort()

            self.dataset_tr = OrderedDict()
            for i in tr_keys:
                self.dataset_tr[i] = self.dataset['train'][i]
            
            # validation set and test set use original image
            self.dataset_val = None
            self.dataset_ts = None

                
    def plot_progress(self):
        try:
            font = {'weight': 'normal',
                    'size': 18}

            matplotlib.rc('font', **font)       

            fig = plt.figure(figsize=(30, 24))   
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            x_values = list(range(self.epoch + 1))

            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")

            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")

            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics, color='g', ls='--', label="evaluation metric")

            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)
            # save progress fig
            fig.savefig(join(self.output_folder, "progress.png"))
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())

    def plot_network_architecture(self):
        """
        can be implemented (see nnUNetTrainer) but does not have to. Not implemented here because it imposes stronger
        assumptions on the presence of class variables
        """      
        pass
    
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        """logging args and starting time.

        Args:
            also_print_to_console (bool, optional): _description_. Defaults to True.
            add_timestamp (bool, optional): _description_. Defaults to True.
        """
        timestamp = time.time()
        dt_time = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_time, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
            self.log_file = join(self.output_folder, f"training_log_{timestamp}.txt")
            with open(self.log_file, 'w') as f:
                f.write("Training Start... \n")

        # Only try 5 times for write log
        successful = False
        max_attempt = 5
        ctr = 0
        while not successful and ctr < max_attempt:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print(f"{datetime.fromtimestamp(timestamp)}:failed to log: ", sys.exc_info())
                time.sleep(0.5)
                ctr += 1

        if also_print_to_console:
            print(*args)

    # checkpoint related
    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time.time()
        model_state_dict = self.network.state_dict()  # Save only the parameters without saving the network structure
        for key in model_state_dict.keys():
            model_state_dict[key] = model_state_dict[key].cpu()
        lr_sche_state_dict = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'state_dict'):
            lr_sche_state_dict = self.lr_scheduler.state_dict()

        optimizer_state_dict = self.optimizer.state_dict() if save_optimizer else None
        # saving start
        self.print_to_log_file("Saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sche_state_dict,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics),
            'best_stuff': (
                self.best_epoch_based_MA_tr_loss, self.best_MA_tr_loss_for_patience, 
                self.best_epoch_based_MA_val_eval_criterion, self.best_MA_val_eval_criterion_for_patience, 
                self.best_val_eval_criterion_MA
            )
        }
        # HACK: 混合精度训练还不懂
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("Saving completed, took %.2f seconds" % (time.time()-start_time))

    def load_checkpoint_ram(self, checkpoint, train=True, pretrained=False):
        """used for if the checkpoint is already in ram

        Args:
            checkpoint (_type_): checkpoint loaded into ram.
            train (bool, optional): _description_. Defaults to True.
        """
        # HACK: why
        if not self.was_initialized:
            self.initialize(train)
        # HACK：当使用多GPU训练的时候回来添加，这时self.network只需要网络的参数

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        if pretrained:
            # 如果想用之前训练好的且接着训练，那就不能装载epoch等训练参数
            pretrain_checpoint = checkpoint["state_dict"]   # 一般参数名
            # pretrain_checpoint = checkpoint   # Segformer的mit_b0
            # pretrain_checpoint = checkpoint["model_state_dict"]
            # 筛选相同名字的参数值
            model_dict = self.network.state_dict()
            state_dict = {k:v for k,v in pretrain_checpoint.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.network.load_state_dict(model_dict)
        else:
            pretrain_checpoint = checkpoint["model_state_dict"]
            model_dict = self.network.state_dict()
            state_dict = {k:v for k,v in pretrain_checpoint.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.network.load_state_dict(model_dict)
            # self.network.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            if train:
                # load optimizer parameters (if present)
                # The parameters of optimizer are not always saved, so we need to import it first to check if it is None
                optimizer_state_dict = checkpoint['optimizer_state_dict']
                if optimizer_state_dict is not None:
                    self.optimizer.load_state_dict(optimizer_state_dict)

                if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and \
                    checkpoint['lr_scheduler_state_dict'] is not None:
                    self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                
                if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                    self.lr_scheduler.step(self.epoch)
            
            self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint['plot_stuff']
            # load best loss (if present)
            if 'best_stuff' in checkpoint.keys():
                self.best_epoch_based_MA_tr_loss, self.best_MA_tr_loss_for_patience, \
                self.best_epoch_based_MA_val_eval_criterion, self.best_MA_val_eval_criterion_for_patience, self.best_val_eval_criterion_MA = \
                    checkpoint['best_stuff']
            
            # after the training is done, the epoch is incremented one more time in nnUNet's old code. This results
            # in self.epoch = 1001 for old trained models when the epoch is actually 1000. It results that len(self.all_tr_losses)
            # not equal self.epoch and plto function will fail.
            if self.epoch != len(self.all_tr_losses):
                self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is \
                        due to an old bug and should only appear when you are loading old models. New models should only \
                        have this fixed! self.epoch is now set to len(self.all_tr_losses)")
                self.epoch = len(self.all_tr_losses)
                self.all_tr_losses = self.all_tr_losses[:self.epoch]
                self.all_val_losses = self.all_val_losses[:self.epoch]
                self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
                self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]
            
        self._maybe_init_amp()

    def load_checkpoint(self, fname, train=True, pretrained=False):
        self.print_to_log_file("Loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # tricks for me
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train, pretrained)

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        # HACK：后缀为.model是啥意思
        if isfile(join(self.output_folder, 'model_best.model')):
            self.load_checkpoint(join(self.output_folder, 'model_best.model'), train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling \
                back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        if isfile(join(self.output_folder, "model_final_checkpoint.model")):
            return self.load_checkpoint(join(self.output_folder, "model_final_checkpoint.model"), train=train)
        if isfile(join(self.output_folder, "model_latest.model")):
            return self.load_checkpoint(join(self.output_folder, "model_latest.model"), train=train)
        if isfile(join(self.output_folder, "model_best.model")):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")    

    def load_final_checkpoint(self, train=False):
        filename = join(self.output_folder, "model_final_checkpoint.model")
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None and torch.cuda.is_available():
            self.amp_grad_scaler = GradScaler() # HACK: 看看相关知识

    def run_training(self):
        _ = self.tr_gen
        _ = self.val_gen

        # 清空缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._maybe_init_amp()

        maybe_mkdir_p(self.output_folder)
        self.plot_network_architecture()

        if cudnn.benchmark and cudnn.deterministic:
            warn("torch.backends.cudnn.deterministic is True indicating a deterministic training is desired. \
                But torch.backends.cudnn.benchmark is True as well and this will prevent deterministic training! \
                If you want deterministic then set benchmark=False")

        # initialize
        if not self.was_initialized:
            self.initialize(True)

        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch + 1)
            epoch_start_time = time.time()
            train_losses_epoch = []

            """train one epoch"""
            self.network.train()

            # progress bar (use tqdm)
            if self.use_progress_bar:
                tbar = tqdm(self.tr_gen)
                for datas in tbar:
                    tbar.set_description(f"Epoch {self.epoch + 1}/{self.max_num_epochs}")

                    l = self.run_iteration(datas, True)

                    tbar.set_postfix(loss=l)
                    train_losses_epoch.append(l)
            else:
                for datas in self.tr_gen:
                    l = self.run_iteration(datas, True)
                    train_losses_epoch.append(l)
                    
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])

            """Validation"""
            with torch.no_grad():

                # train = False
                self.network.eval()
                val_losses = []
                if not self.ts_mul:
                    for datas in self.val_gen:
                        l = self.run_iteration(datas, False, True, need_upsample=self.need_upsample)
                        val_losses.append(l)
                else:
                    l_sum = []
                    # data_counts = 0
                    for _, datas_val in self.val_gen.items():
                        for datas in datas_val:
                            l_ = self.run_iteration(datas, False, True, need_upsample=self.need_upsample)
                            l_sum.append(l_)
                        val_losses.append(np.mean(l_sum))
                        # # 由于使用多个测试集, 我们需要求所有数据集平均指标的均值
                        # temp_online_eval_foreground_dc = self.online_eval_foreground_dc[:data_counts]
                        # temp_online_eval_foreground_dc.append(np.mean(self.online_eval_foreground_dc[data_counts:]))
                        # self.online_eval_foreground_dc = temp_online_eval_foreground_dc
                        # data_counts += 1


                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("validation loss: %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for datas in self.val_gen:
                        l = self.run_iteration(datas, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("validation loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])

            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training

            continue_training = self.on_epoch_end()
            epoch_end_time = time.time()

            if not continue_training:
                # allows for early stopping
                break
            
            self.epoch += 1
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.epoch -= 1 # if we don't do this we can get a problem with loading model_final_checkpoint.

        if self.save_final_checkpoint: self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False, need_upsample=False):
        """iterate training

        Args:
            data_generator (_type_): _description_
            do_backprop (bool, optional): to distinguish training and inference process. Defaults to True (training).
            run_online_evaluation (bool, optional): _description_. Defaults to False.
        """
        if need_upsample:
            data, target, img_size, _ = data_generator
        
        else:
            data, target = data_generator

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
        
        self.optimizer.zero_grad()

        if self.fp16:
            # Casts operations to mixed precision
            with autocast():
                output = self.network(data)
                # self.gpu_track.track()
                if need_upsample:
                    output = F.interpolate(output, size=img_size, mode='bilinear', align_corners=False)
                del data
                l1 = self.iou_loss(output, target)
                l2 = self.dice_loss(output, target)
                l = l1 + l2
                # self.gpu_track.track()
                # l = self.loss(output, target)

            if do_backprop:
                # Scales the loss, and calls backward()
                # to create scaled gradients
                self.amp_grad_scaler.scale(l).backward()
                # Unsacles gradients and calls optimizer.step() if gradients are not inf or NaN
                # or skip optimizer.step() to ensure that weights are not updated
                self.amp_grad_scaler.step(self.optimizer)
                # Updates the scale for next iteration
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()   

    def run_online_evaluation(self, *args, **kwargs):
        """Can be implemented, does not have to
        """
        pass

    def finish_online_evaluation(self):
        """Can be implemented, does not have to
        """
        pass

    """training utils"""
    def on_epoch_end(self):
        self.finish_online_evaluation() # does not have to do anything, but can be used to update self.all_val_eval_metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_trainging = self.manage_patience()
        return continue_trainging 

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_MA_val_eval_criterion_for_patience = self.val_eval_criterion_MA

            if self.best_epoch_based_MA_tr_loss is None:
                self.best_epoch_based_MA_tr_loss = self.epoch
                self.best_epoch_based_MA_val_eval_criterion = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                # NOTE: 检查val上的metric是否超过现有最好metric, 如果是, 则更新metric及metric所在epoch
                self.best_MA_val_eval_criterion_for_patience = self.val_eval_criterion_MA
                self.best_epoch_based_MA_val_eval_criterion = self.epoch
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model")) 

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_MA_tr_loss = self.epoch

            else:
                pass

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_MA_tr_loss > self.patience:
                if self.optimizer.params_groups[0]['lr'] > self.lr_threshold:
                    self.best_epoch_based_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    continue_training = False

            # NOTE: 加多一个判断, 判断val上的metric是否已经self.patience轮没有更新了, 如果是, 则early stop(这里不判断最低学习率)
            elif self.epoch - self.best_epoch_based_MA_val_eval_criterion > self.patience:
                continue_training = False
            else:
                pass
        return continue_training

    def maybe_update_lr(self):
        """maybe update learning rate
        """
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (
                lr_scheduler.ReduceLROnPlateau, lr_scheduler.StepLR, lr_scheduler.CosineAnnealingLR, _LRScheduler
                ))

            if isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau)):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)

            elif self.cfg.training.lr_scheduler == "StepLR":
                # 由于构建时使用GradualWarmupScheduler, 因此无法识别
                self.lr_scheduler.step(self.epoch + 1)

            elif isinstance(self.lr_scheduler, lr_scheduler.CosineAnnealingLR):
                self.lr_scheduler.step()
                
            self.print_to_log_file(f"lr is now (scheduler) {str(self.optimizer.param_groups[0]['lr'])}")

    def maybe_save_checkpoint(self):
        """Saves a checkpoint every save_ever epochs
        """
        if self.save_intermediate_checkpoints and (self.epoch % self.save_every == (self.save_every - 1)):
            self.print_to_log_file("Saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")    

    def update_train_loss_MA(self):
        """Moving average loss
        """
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                # If it is loss, then the the lower the better
                # If it is some indicator, then the higher the better
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - \
                    (1 - self.val_eval_criterion_alpha) * self.all_val_losses[-1]
            else:
                # self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + \
                    (1 - self.val_eval_criterion_alpha) * self.all_val_eval_metrics[-1]

    @abstractmethod
    def validate(self, *args, **kwargs):
        pass

    def find_lr(self, num_iters=1000, init_value=1e-6, final_value=10., beta=0.98):
        """stolen and adapted from here: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html

        Args:
            num_iters (int, optional): _description_. Defaults to 1000.
            ini_value (_type_, optional): _description_. Defaults to 1e-6.
            final_value (_type_, optional): _description_. Defaults to 10..
            beta (float, optional): _description_. Defaults to 0.98.
        """
        import math
        self._maybe_init_amp()
        mult = (final_value / init_value) ** (1 / num_iters)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        for batch_num in range(1, num_iters + 1):
            # +1 because this one here is not designed to have negative loss...
            loss = self.run_iteration(self.tr_gen, do_backprop=True, run_online_evaluation=False).data.item() + 1

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        lrs = [10 ** i for i in log_lrs]
        fig = plt.figure()
        plt.xscale('log')
        plt.plot(lrs[10:-5], losses[10:-5])
        plt.savefig(join(self.output_folder, "lr_finder.png"))
        plt.close()
        return log_lrs, losses
