"""
❤Descripttion: your project
❤version: 1.0
❤Author: MilknoCandy
❤Date: 2022-11-30 16:14:27
❤LastEditTime: 2023-02-26 09:30:30
❤Github: https://github.com/MilknoCandy
"""
import random
import time
from collections import OrderedDict

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from alive_progress import alive_bar
from batchgenerators.utilities.file_and_folder_operations import *
from loguru import logger
from medpy.metric.binary import assd, hd95
from monai.metrics import (compute_meandice, compute_meaniou,
                           compute_surface_dice)
from monai.metrics.confusion_matrix import (compute_confusion_matrix_metric,
                                            get_confusion_matrix)
from ptflops import get_model_complexity_info
from tabulate import tabulate
from thop import profile
from torch.utils.data import DataLoader
from torchinfo import summary

from network_architecture.network_tools.build_lr_schedulers import \
    build_lr_scheduler
from network_architecture.network_tools.build_model import build_network
from network_architecture.network_tools.build_optimizers import build_optimizer
from network_architecture.simImple import softmax_helper
from training.data_loading.dataset_loading import (CustomDataSet, TestDataSet,
                                                   load_dataset)
from training.learning_rate.cosine_lr import cosine_scheduler
from training.learning_rate.poly_lr import poly_scheduler
from training.loss_function.dice_loss import DiceLoss, dice_bce_loss
from training.loss_function.iou_loss import weighted_IoU_and_BCE
from training.params_initialization.initialization import InitWeights_He
from training.trainer.network_trainer import NetworkTrainer
from utils.featmaps_visualize.grad_cam_visualize import gradcam_visualize
from utils.featmaps_visualize.inter_featmap_visulaize import featmap_visualize
from utils.featmaps_visualize.unecrtainty_visulaize import entropy_visualize
from utils.gpu_track.gpu_mem_track import MemTracker
from utils.metrics.compute_FPS import compute_speed
from utils.metrics.compute_metrics import AllMterics
from utils.tensor_utilities import sum_tensor
from utils.to_torch import maybe_to_torch, to_cuda


class CANetTrainer(NetworkTrainer):
    def __init__(
        self, cfg=None, data_type='0', output_folder=None, dataset_directory=None, sed=1, deterministic=True, fp16=False, fold='all'
    ):
        super().__init__(data_type, output_folder, dataset_directory, sed, deterministic, fp16)
        
        self.experiment_name = self.__class__.__name__
        
        self.cfg = cfg
        self.max_num_epochs = cfg.training.max_num_epochs
        self.warm_up_epochs = cfg.training.warm_up_epochs

        self.initial_lr = cfg.training.initial_lr
        self.weight_decay = cfg.training.weight_decay
        self.lr_scheduler_eps = cfg.training.lr_scheduler_eps
        self.lr_scheduler_patience = cfg.training.lr_scheduler_patience

        self.pin_memory = True
        self.fold = fold
        self.dsplit = cfg.dataset.dsplit
        self.batch_size = cfg.dataset.batch_size
        self.crop_sizeH = cfg.dataset.crop_size.h
        self.crop_sizeW = cfg.dataset.crop_size.w
        self.val_eval_criterion_alpha = 0.1     # 用于平滑验证集的metric

        self.ts_use_npz = cfg.dataset.test_use_npz      # 为True则使用resize后的npz文件进行测试; False则使用原图像进行测试
        self.need_upsample = not self.ts_use_npz
        # test数据使用原图像数据, 因此img和seg是分开存储的, 从配置文件cfg获取
        self.ts_img_pth = cfg.dataset.test_img
        self.ts_lab_pth = cfg.dataset.test_lab
        self.all_size = cfg.dataset.all_size
        # 是否为多类分割
        self.num_classes = cfg.dataset.num_classes
        self.mask_onehot = False if self.num_classes==1 else True
    
        torch.set_num_threads(8)
        # # track memory
        # self.gpu_track = MemTracker()
        
    def initialize(self, training=True):
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            # self.loss = dice_bce_loss
            self.iou_loss = weighted_IoU_and_BCE
            self.dice_loss = DiceLoss(dims=(2,3))

            if training:
                self.tr_gen, self.val_gen = self.get_basic_generators()
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())), also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())), also_print_to_console=False)
            
            else:
                pass
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()
        else:
            self.print_to_log_file("self.was_initialized is True, not running self.initialize again")
        self.was_initialized = True

    def initialize_network(self):
        """
        - SGD or Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep_supervision is True, reflected in TransFuse's Loss

        Returns:
            _type_: _description_
        """
        self.network = build_network(model_name=self.cfg.model.model_name, num_classes=self.num_classes)

        total = sum([param.nelement() for param in self.network.parameters()])
        print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))
        
        if torch.cuda.is_available():
            self.network.cuda()
            # self.gpu_track.track()


    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = build_optimizer(
            self.cfg.training.optimizer, self.network.parameters(), self.initial_lr, self.weight_decay
        )
        if self.cfg.training.lr_scheduler is not None:
            self.lr_scheduler = build_lr_scheduler(
                self.cfg.training.lr_scheduler, self.optimizer, self.lr_scheduler_patience, self.lr_scheduler_eps
            )
        
    def plot_network_architecture(self):
        try:
            # import hiddenlayer as hl
            # from batchgenerators.utilities.file_and_folder_operations import \
            #     join
            # if torch.cuda.is_available():
            #     g = hl.build_graph(self.network, torch.rand((1, 3, self.crop_sizeH, self.crop_sizeW)).cuda(),
            #                        transforms=None)
            # else:
            #     g = hl.build_graph(self.network, torch.rand((1, 3, self.crop_sizeH, self.crop_sizeW)),
            #                        transforms=None)
            # g.save(join(self.output_folder, "network_architecture.pdf"))
            # del g
            # self.print_to_log_file("\nprinting the network instead:\n")
            # self.print_to_log_file(self.network)
            if torch.cuda.is_available():
                input = torch.randn((1, 3, self.crop_sizeH, self.crop_sizeW)).cuda()
            else:
                input = torch.randn((1, 3, self.crop_sizeH, self.crop_sizeW))
            macs, params = profile(self.network, inputs=(input, ))
            logger.info(f"| model |macs:', {macs/1e9}, 'params:', {params/1e6}|")
            flops, params = get_model_complexity_info(self.network, (3, self.crop_sizeH, self.crop_sizeW))
            # logger.info(f"| model |flops:', {flops/1e9}, 'params:', {params/1e6}|")
            self.print_to_log_file(flops)
            self.print_to_log_file(params)
            self.print_to_log_file(summary(self.network, input_size=(1, 3, self.crop_sizeH, self.crop_sizeW), depth=5)) # for torchinfo
            self.print_to_log_file("\n")

        except Exception as e:
            self.print_to_log_file("Unable to plot network architecture:")
            self.print_to_log_file(e)

            self.print_to_log_file("\nprinting the network instead:\n")
            self.print_to_log_file(self.network)
            # self.print_to_log_file(summary(self.network, input_size=(1, 3, self.crop_sizeH, self.crop_sizeW), depth=5)) # for torchinfo
            self.print_to_log_file("\n")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def load_dataset(self):
        self.dataset = load_dataset(self.dataset_directory, data_type=self.data_type, data_suffix="npz")
        
    def get_basic_generators(self):
        self.load_dataset()
        self.do_split(dsplit=self.dsplit)
        tr_gen = CustomDataSet(self.dataset_tr, transform_list=self.cfg.Train_transform_list, oneHot=self.mask_onehot)
        tr_dl = DataLoader(tr_gen, batch_size=self.batch_size, shuffle=True, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)
        if not self.ts_use_npz:
            val_gen = TestDataSet(
                self.dataset_val, 
                self.val_size, 
                crop_size=(self.crop_sizeH, self.crop_sizeW), 
                transform_list=self.cfg.Val_transform_list, oneHot=self.mask_onehot
            )
            val_dl = DataLoader(val_gen, batch_size=1, shuffle=False, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)
        else:
            val_gen = CustomDataSet(self.dataset_val, transform_list=self.cfg.Val_transform_list, oneHot=self.mask_onehot)
            val_dl = DataLoader(val_gen, batch_size=1, shuffle=False, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)

        return tr_dl, val_dl

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        Args:
            epoch (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        super().maybe_update_lr()
        if self.lr_scheduler is None:
            if epoch is None:
                ep = self.epoch + 1
            else:
                ep = epoch
            # self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
            # self.optimizer.param_groups[0]['lr'] = poly_scheduler(base_value=self.initial_lr, final_value=self.lr_threshold, epoch=ep, 
            #         max_epochs=self.max_num_epochs, warmup_epochs=self.warm_up_epochs, start_warmup_value=1e-6)
            self.optimizer.param_groups[0]['lr'] = cosine_scheduler(base_value=self.initial_lr, final_value=self.lr_threshold, epoch=ep, 
                    max_epochs=self.max_num_epochs, warmup_epochs=self.warm_up_epochs, start_warmup_value=1e-6)
            self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        continue_training = super().on_epoch_end()
        # continue_training = self.epoch < self.max_num_epochs    # 重置continue_training, 使得网络一直训练至最大轮数(耗时太长)

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training
        
    def run_training(self):

        dct = OrderedDict()
        for k in self.__dir__():
            if not k.startswith("__"):
                if not callable(getattr(self, k)):
                    dct[k] = str(getattr(self, k))
        del dct['dataset']
        del dct['dataset_tr']
        del dct['dataset_val']
        del dct['dataset_ts']
        save_json(dct, join(self.output_folder, "debug.json"))
        # self.maybe_update_lr(self.epoch)
        super().run_training()


    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value ang the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        将每个 batch 当作一张图像求 dice , 迭代一个 epoch 后, 再对每个 batch 的 dice 求平均(仅是一个估计值, 不准确)

        Args:
            output (tensor): prediction
            target (tensor): groundtruth
        """
        # output: (batch × num_classes × H × W( × D))
        with torch.no_grad():
            num_classes = output.shape[1]
            if num_classes==1:
                output_sigmoid = output.sigmoid()
                output_seg = 1*(output_sigmoid>0.5)   # 单标签分割
                output_seg = output_seg[:, 0]
                target = target[:, 0]
                num_classes += 1
            else:
                output_softmax = F.softmax(output, 1)
                output_seg = output_softmax.argmax(1)   # 多标签分割
                target = target.argmax(1)   # b×num×h×w

            
            axes = tuple(range(1, len(target.shape)))
            # nnunet直接是num_classes - 1
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            # for c in range(0, num_classes):   # nnunet直接是num_classes/
            for c in range(1, num_classes):   # nnunet直接是num_classes
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))
    
    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        # self.all_val_eval_metrics.append(np.mean(global_dc_per_class))
        self.all_val_eval_metrics.append(np.mean(self.online_eval_foreground_dc))

        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.Because it consider every batch as one image)")
        self.print_to_log_file("real foreground Dice:", np.mean(self.online_eval_foreground_dc))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
    ############################ ENTROPY_VISUALIZE ##############################
    def uncertain_visualize(self, output_file: str="uncertain_visualize", gt_oneHot=False):
        save_folder = join(self.output_folder, output_file)

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

        if self.dataset_ts is None:
            self.load_dataset()
            self.do_split()

        if not self.ts_use_npz:
            ts_gen = TestDataSet(
                self.dataset_ts, 
                self.ts_size, 
                crop_size=(self.crop_sizeH, self.crop_sizeW), 
                transform_list=self.cfg.Test_transform_list, 
                oneHot=gt_oneHot
            )
        else:
            ts_gen = CustomDataSet(self.dataset_ts, transform_list=self.cfg.Test_transform_list, oneHot=gt_oneHot)
        ts_dl = DataLoader(ts_gen, batch_size=1, shuffle=False, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)
        entropy_visualize(self.network, save_folder, ts_dl, (self.crop_sizeH, self.crop_sizeW))

    ############################ HotMap_VISUALIZE ##############################
    def featmap_vis(self, output_file: str="featmap_visualize", method=None, gt_oneHot=False, layer_aggregate=False):
        save_folder = join(self.output_folder, output_file)

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

        if self.dataset_ts is None:
            self.load_dataset()
            self.do_split()

        if not self.ts_use_npz:
            ts_gen = TestDataSet(
                self.dataset_ts, 
                self.ts_size, 
                crop_size=(self.crop_sizeH, self.crop_sizeW), 
                transform_list=self.cfg.Test_transform_list, 
                oneHot=gt_oneHot
            )
        else:
            ts_gen = CustomDataSet(self.dataset_ts, transform_list=self.cfg.Test_transform_list, oneHot=gt_oneHot)
        ts_dl = DataLoader(ts_gen, batch_size=1, shuffle=False, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)
        if method is None:
            featmap_visualize(self.network, save_folder, ts_dl, (self.crop_sizeH, self.crop_sizeW))
        else:
            gradcam_visualize(self.network, save_folder, ts_dl, (self.crop_sizeH, self.crop_sizeW), method=method, layer_aggregate=layer_aggregate)

    def output_seg(self, save_output_folder: str="Test_result", gt_oneHot=False):
        self.network.eval()
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

        if self.dataset_ts is None:
            self.load_dataset()
            self.do_split(self.dsplit)
        
        ts_folder = join(self.output_folder, save_output_folder)
        maybe_mkdir_p(ts_folder)
        
        if not self.ts_use_npz:
            ts_gen = TestDataSet(
                self.dataset_ts, 
                self.ts_size, 
                crop_size=(self.crop_sizeH, self.crop_sizeW), 
                transform_list=self.cfg.Test_transform_list, 
                oneHot=gt_oneHot
            )
        else:
            ts_gen = CustomDataSet(self.dataset_ts, transform_list=self.cfg.Test_transform_list, oneHot=gt_oneHot)
        ts_dl = DataLoader(ts_gen, batch_size=1, shuffle=False, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)

        self.print_to_log_file("=============================Outputing Segmentation=============================")

        data_nums = len(ts_dl)
        
        with alive_bar(len(ts_dl), title="Testing", bar='halloween', spinner='loving', length=80, force_tty=True) as bar:
            for i, data in enumerate(ts_dl):

                if not self.ts_use_npz:
                    input, gt, img_size, imgo = data
                else:
                    input, gt = data
                input = maybe_to_torch(input)
                gt = maybe_to_torch(gt)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    input = to_cuda(input)
                    gt = to_cuda(gt)
                    
                with torch.no_grad():
                    pred = self.network(input)
                num_classes = pred.shape[1]
                if num_classes==1:
                    if not self.ts_use_npz:
                        pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False)
                    pred = torch.sigmoid(pred)
                    pred = 1.0*(pred>0.5)

                else:
                    if not self.ts_use_npz:
                        pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False)
                    pred = softmax_helper(pred)
                    pred = pred.argmax(1)
                    if gt_oneHot:
                        gt = gt.argmax(1)

                pred_mask = pred.squeeze().cpu().numpy()*255
                mask = gt.squeeze().cpu().numpy()*255
                imgo_ = imgo.squeeze().cpu().numpy()

                # imgo_ = cv2.cvtColor(imgo_, cv2.COLOR_BGR2RGB)

                # # 阈值处理为二值图像
                # # _, thresh_pred = cv2.threshold(np.uint8(pred_mask), 127, 255, 0)
                # _, thresh_pred = cv2.threshold(np.uint8(pred_mask), 127, 255, cv2.THRESH_BINARY)
                # _, thresh_mask = cv2.threshold(np.uint8(mask), 127, 255, 0)
                # # 绘制轮廓
                # thick_line = round(3/1024*max(imgo_.shape))
                # contours_mask, _ = cv2.findContours(thresh_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # contours_pred, _ = cv2.findContours(thresh_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # cnt = contours_mask[0]
                # cv2.drawContours(imgo_, [cnt], -1, (0, 255, 255), thick_line)
                # cnt = contours_pred[0]
                # cv2.drawContours(imgo_, [cnt], -1, (0, 255, 0), thick_line)
                # save_here = join(ts_folder, f"{i+1}_combine.png")
                # cv2.imwrite(save_here, imgo_)
                
                # 输出二值图像
                save_here = join(ts_folder, f"{i+1}_prediction.png")
                imageio.imwrite(save_here, np.uint8(pred_mask))
                bar()


        self.print_to_log_file("===================================DONE=========================================")
        
    def validate(self, test_output_folder: str="Test_result", gt_oneHot=False):
        current_mode = self.network.training
        self.network.eval()

        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"

        if self.dataset_ts is None:
            self.load_dataset()
            self.do_split(self.dsplit)
        
        ts_folder = join(self.output_folder, test_output_folder)
        maybe_mkdir_p(ts_folder)
        

        if not self.ts_use_npz:
            ts_gen = TestDataSet(
                self.dataset_ts, 
                self.ts_size, 
                crop_size=(self.crop_sizeH, self.crop_sizeW), 
                transform_list=self.cfg.Test_transform_list, 
                oneHot=gt_oneHot
            )
        else:
            ts_gen = CustomDataSet(self.dataset_ts, transform_list=self.cfg.Test_transform_list, oneHot=gt_oneHot)
        ts_dl = DataLoader(ts_gen, batch_size=1, shuffle=False, num_workers=self.cfg.dataset.num_workers, pin_memory=self.pin_memory)

        all_metrics = AllMterics()
        # logger.info("\n=============================Testing=============================")
        self.print_to_log_file("=============================Testing=============================")
        start_time = time.time()
        with alive_bar(len(ts_dl), title="Testing", bar='halloween', spinner='loving', length=80, force_tty=True) as bar:
            for i, data in enumerate(ts_dl):
                if not self.ts_use_npz:
                    input, gt, img_size, _ = data
                else:
                    input, gt = data
                input = maybe_to_torch(input)
                gt = maybe_to_torch(gt)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    input = to_cuda(input)
                    gt = to_cuda(gt)

                with torch.no_grad():
                    pred = self.network(input)
                num_classes = pred.shape[1]
                if num_classes==1:
                    if not self.ts_use_npz:
                        pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False)
                    pred = torch.sigmoid(pred)
                    pred = 1.0*(pred>0.5)

                else:
                    if not self.ts_use_npz:
                        pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False)
                    pred = softmax_helper(pred)
                    pred = pred.argmax(1)
                    if gt_oneHot:
                        gt = gt.argmax(1)

                pred_mask = pred.cpu()[None, ...]
                mask = gt.cpu()

                all_metrics.refresh(pred_mask, mask)
                
                bar()
            
            end_time = time.time()
            table_header = ['Dataset', 'Model_name', 'mDice', 'mIoU', 'mAcc', 'TJI', 'sen', 'spe', 'precision','Time Consumption']
            table_data = [(
                'ISIC 2018', self.cfg.model.model_name, 
                all_metrics.dice.avg, 
                all_metrics.jc.avg, 
                all_metrics.acc.avg, 
                all_metrics.tji.avg,
                all_metrics.sen.avg,
                all_metrics.spe.avg,
                all_metrics.prec.avg,
                end_time-start_time
            )]
            self.print_to_log_file('\n', tabulate(table_data, headers=table_header,tablefmt='psql'))

    def eval_speed(self, input_size=(1, 3, 512, 512)):
        print("=============================Testing Speed=============================")
        compute_speed(self.network, input_size=input_size)