import os
import copy
import time
import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from lib.models import Generic_UNet
from lib.datasets import BGDataset, get_moreDA_augmentation
from lib.configs.parse_arg import opt, args
from lib.utils import MultiLossMeter,  CELoss2, DiceMetric, NpEncoder
from lib.vis.save_inference import save_pred
import ast
import json
import shutil
import gc
import matplotlib as mpl;

mpl.use('Agg')


class NLSS(object):
    def __init__(self):
        super(NLSS, self).__init__()
        self.since = time.time()
        self.phases = []
        self.val_ids = None
        self.save_train_ids = None

        # EM style
        self.EM_save = False
        if opt.framework.E_step_epoch >= 0 or opt.framework.Auto_E_step_epoch:
            self.E_train_loader = None
            self.save_val_ids = None
            self.EM_save = True

        self.data_loaders = self.build_dataloader()
        self.model = self.build_model()

        if opt.framework.co_teaching:
            self.model2 = self.build_model(model2=True)
            self.best2 = {'loss': 1e8, 'dice': 0, 'epoch': -1}

        if opt.framework.tri_net:
            self.model3 = self.build_model(model3=True)
            self.best3 = {'loss': 1e8, 'dice': 0, 'epoch': -1}

        self.criterion, self.metric, self.scheduler, self.optimizer = self.build_optimizer()

        self.best = {'loss': 1e8, 'dice': 0, 'epoch': -1}
        self.log_init()
        if (args.demo == ''):
            self.writer = SummaryWriter(opt.tb_dir)
        self.save_pred = save_pred

    def build_dataloader(self):
        data_dir = opt.data.data_dir
        data_name = opt.data.dataname

        # training phase
        if args.demo == '':
            self.phases = ['train', 'val']
            if opt.data.dataset == 'BGDataset':
                # dataset
                ds_train = BGDataset(data_dir, phase='train', cls=data_name, aug_rot90=opt.data.aug_rot90,
                                     batch_size=opt.train.train_batch, shuffle=True)
                ds_val = BGDataset(data_dir, phase='val', cls=data_name,
                                   batch_size=opt.train.valid_batch, shuffle=False)

                patch_size = (next(ds_val))['data'].shape[-2:]

                # dataloader
                pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
                if opt.model.deep_supervision:
                    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                        np.vstack(pool_op_kernel_sizes), axis=0))[:-1]
                else:
                    deep_supervision_scales = None
                extra_label_keys = None
                if opt.data.load_clean_label or opt.data.load_superpixel_label:
                    extra_label_keys = []
                    if opt.data.load_clean_label:
                        extra_label_keys.append('clean_label')
                    if opt.data.load_superpixel_label:
                        extra_label_keys.append('superpixel')
                train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size, \
                                                                   deep_supervision_scales=deep_supervision_scales,
                                                                   seeds_train=args.seed, seeds_val=args.seed,
                                                                   pin_memory=True,
                                                                   extra_label_keys=extra_label_keys,
                                                                   extra_only_train=True)
                data_loaders = {
                    'train': train_loader,
                    'val': val_loader,
                }

                # load extra dataloader for saving pseudo labels
                if self.EM_save:
                    valid_batch = 200  # set batch=1 for saving each image
                    if opt.data.dataname != 'lesion':
                        valid_batch = 197
                    ds_train = BGDataset(data_dir, phase='train', cls=data_name, batch_size=valid_batch, shuffle=False)
                    ds_val = BGDataset(data_dir, phase='val', cls=data_name, batch_size=valid_batch, shuffle=False)
                    self.E_train_loader, _ = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size,
                                                                     deep_supervision_scales=deep_supervision_scales,
                                                                     seeds_train=args.seed, seeds_val=args.seed,
                                                                     pin_memory=True, val_mode=True,
                                                                     extra_label_keys=extra_label_keys,
                                                                     extra_only_train=True)

                    self.save_val_ids = {'train': ds_train.samples}

        # inference phase
        else:
            # phases
            if ',' in args.demo:
                [self.phases.append(phase.strip()) for phase in (args.demo).split(',')]
                print(self.phases)
            else:
                self.phases = [(args.demo).strip()]

            if opt.data.dataset == 'BGDataset':
                # dataset
                ds_train, ds_val = None, None
                self.val_ids = {}
                valid_batch = 1  # 32, opt.train.valid_batch
                if 'val' in self.phases:
                    ds_val = BGDataset(data_dir, phase='val', cls=data_name,
                                       batch_size=valid_batch, shuffle=False)
                    self.val_ids['val'] = ds_val.samples
                    patch_size = (next(ds_val))['data'].shape[-2:]
                if 'test' in self.phases:
                    ds_test = BGDataset(data_dir, phase='test', cls=data_name,
                                        batch_size=valid_batch, shuffle=False)
                    self.val_ids['test'] = ds_test.samples
                    patch_size = (next(ds_test))['data'].shape[-2:]
                if 'train' in self.phases:
                    if args.save_select:
                        valid_batch = opt.train.train_batch
                        print('use train batch size', valid_batch)
                    ds_train = BGDataset(data_dir, phase='train', cls=data_name,
                                         batch_size=valid_batch, shuffle=False)
                    self.val_ids['train'] = ds_train.samples
                    patch_size = (next(ds_train))['data'].shape[-2:]

                # dataloader
                pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
                if opt.model.deep_supervision:
                    deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
                        np.vstack(pool_op_kernel_sizes), axis=0))[:-1]
                else:
                    deep_supervision_scales = None
                extra_label_keys = None
                if opt.data.load_clean_label or opt.data.load_superpixel_label:
                    extra_label_keys = []
                    if opt.data.load_clean_label:
                        extra_label_keys.append('clean_label')
                    if opt.data.load_superpixel_label:
                        extra_label_keys.append('superpixel')
                train_loader, val_loader = get_moreDA_augmentation(ds_train, ds_val, patch_size=patch_size,
                                                                   deep_supervision_scales=deep_supervision_scales,
                                                                   seeds_train=args.seed, seeds_val=args.seed,
                                                                   pin_memory=True, val_mode=True,
                                                                   extra_label_keys=extra_label_keys,
                                                                   extra_only_train=True)
                data_loaders = {'train': train_loader, 'val': val_loader}
                if 'test' in self.phases:
                    _, test_loader = get_moreDA_augmentation(None, ds_test, patch_size=patch_size,
                                                             deep_supervision_scales=deep_supervision_scales,
                                                             seeds_train=args.seed, seeds_val=args.seed,
                                                             pin_memory=True, val_mode=True)
                    data_loaders['test'] = test_loader

        return data_loaders

    def build_model(self, model2=False, model3=False):
        if opt.model.network == 'Generic_UNet':
            # net params
            input_channels = 1
            if opt.model.input_channel == 3:
                input_channels = 3
            self.pool_op_kernel_sizes = ast.literal_eval(opt.model.pool_op_kernel_sizes)
            conv_kernel_sizes = ast.literal_eval(opt.model.conv_kernel_sizes)
            deep_supervision = opt.model.deep_supervision
            net_params = {
                'input_channels': input_channels, 'base_num_features': 32, 'num_classes': 2,
                'num_pool': len(self.pool_op_kernel_sizes),

                'num_conv_per_stage': 2, 'feat_map_mul_on_downscale': 2, 'conv_op': nn.Conv2d,
                'norm_op': nn.BatchNorm2d, 'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                'dropout_op': nn.Dropout2d, 'dropout_op_kwargs': {'p': 0, 'inplace': True},
                'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
                'deep_supervision': deep_supervision, 'dropout_in_localization': False, 'final_nonlin': lambda x: x,

                'pool_op_kernel_sizes': self.pool_op_kernel_sizes,
                'conv_kernel_sizes': conv_kernel_sizes,
                'upscale_logits': False, 'convolutional_pooling': True, 'convolutional_upsampling': True,
            }
            model = Generic_UNet(**net_params)

        # data parallel, load state dict
        if args.parallel:
            model = nn.DataParallel(model)
        if args.demo != '' and not model2:  # load trained model
            weight_path = args.weight_path
            model_dict = torch.load(weight_path)
            model.load_state_dict(model_dict)
        elif opt.model.use_finetune and opt.model.finetune_model_path != '':
            if model2:
                opt.model.finetune_model_path = opt.model.finetune_model_path.replace('model', 'model2')
            if model3:
                opt.model.finetune_model_path = opt.model.finetune_model_path.replace('model', 'model3')
            model_dict = torch.load(opt.model.finetune_model_path)
            model.load_state_dict(model_dict)

        model = model.cuda()
        return model

    def build_optimizer(self):
        # loss criterion
        if opt.train.loss_name == 'MultipleOutputLoss2' or 'CELoss2':
            criterion = CELoss2()

        # metric, optimizer, scheduler
        metric = DiceMetric(dice_each_class=False)  # dice metric

        if opt.framework.tri_net:
            optimizer = torch.optim.SGD(
                list(self.model.parameters()) + list(self.model2.parameters()) + list(self.model3.parameters()),
                lr=opt.train.lr, momentum=opt.train.momentum, weight_decay=opt.train.weight_decay)
        elif opt.framework.co_teaching:
            optimizer = torch.optim.SGD(list(self.model.parameters()) + list(self.model2.parameters()), lr=opt.train.lr, \
                                        momentum=opt.train.momentum, weight_decay=opt.train.weight_decay)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.train.lr, momentum=opt.train.momentum,
                                        weight_decay=opt.train.weight_decay)

        if opt.train.lr_decay == 'multistep':
            milestones = ast.literal_eval(opt.train.milestones)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=opt.train.gamma)
        elif opt.train.lr_decay == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=opt.train.plateau_patience,
                                                                   factor=opt.train.plateau_gamma)
        else:  # {constant, poly}
            scheduler = None

        return criterion, metric, scheduler, optimizer

    """
    nnUnet Training Procedure.
    """
    def train(self):
        # init
        num_epochs = opt.train.n_epochs
        if (args.max_epoch != -1) and (args.max_epoch < num_epochs):
            num_epochs = args.max_epoch
            print('Early stopping used: max epoch %d' % num_epochs)
        loss_meter, dice_meter = MultiLossMeter(), MultiLossMeter()
        total_iter = {'train': 0, 'val': 0, 'test': 0}
        ## warmup dataloader
        print('Num per train_loader', sum(1 for _ in self.data_loaders['train']))
        print('Num per valid_loader', sum(1 for _ in self.data_loaders['val']))
        ## mean dice of last 10 epochs
        last10_meter = MultiLossMeter()
        last10_meter.reset()

        # epoch
        epoch_notrain = 0  # {0, -1}, no training and loss backward at this epoch
        for epoch in tqdm(range(num_epochs + 1)):

            for phase in self.phases:
                if phase == 'train' or phase == 'train_val':
                    self.model.train(True)
                else:
                    self.model.train(False)
                    sum(1 for _ in self.data_loaders['val'])  # for debugging val_loader error: unexpected number
                loss_meter.reset();
                dice_meter.reset()
                count_batch = 0

                while count_batch < opt.train.n_batches:
                    for i_batch, data in enumerate(self.data_loaders[phase]):
                        inputs = data['image'].float().cuda()
                        if not isinstance(data['gt'], list):
                            data['gt'] = [data['gt']]
                        targets = [it.long().cuda() for it in data['gt']]

                        # forward
                        outputs = self.model(inputs)
                        if not isinstance(outputs, tuple):
                            outputs = tuple([outputs])  # convert tensor to tuple. Direct tuple() not work.
                        preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]  # preds (B, H, W)

                        # loss, backward
                        if phase == 'train' and epoch != epoch_notrain:
                            losses, loss_names = self.criterion.forward(outputs[0], targets[0])
                            loss = losses[0]
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                        else:
                            with torch.no_grad():
                                losses, loss_names = self.criterion.forward(outputs[0], targets[0])
                            loss = losses[0]

                        # loss meter, dice meter
                        loss_meter.update(losses, loss_names)
                        dices, dice_names = self.metric.forward(preds, targets[0])
                        dice_meter.update(dices, dice_names)

                        # plot running loss curve
                        if opt.data.write_log_batch:  # logging per iteration
                            metrics = {'loss': loss, 'dice': dices[0]}
                            total_iter[phase] += 1
                            self.log_batch(metrics, epoch, i_batch)

                        count_batch += 1
                        if count_batch >= opt.train.n_batches:
                            break

                        # release memory
                        del inputs, targets, preds, outputs, losses, dices

                    if phase == 'val':
                        break
                    if phase == 'train' and epoch == epoch_notrain:
                        break

                # epoch metric
                avg_terms = loss_meter.get_metric()
                dice_terms = dice_meter.get_metric()
                avg_terms = {**avg_terms, **dice_terms}
                self.plot_curves_multi(avg_terms, epoch, phase=phase)

                if phase == 'val':
                    ## update last 10 dice meter
                    if epoch > num_epochs - 10:
                        last10_meter.update([avg_terms['dice']], ['dice'])
                self.log_epoch(avg_terms, epoch, phase)
            self.log_time()

            # scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_terms['dice'])
                else:
                    self.scheduler.step()
            elif opt.train.lr_decay == 'poly':
                lr_ = opt.train.lr * (1 - epoch / opt.train.n_epochs) ** 0.9  # power=0.9
                if lr_ < opt.train.min_lr:  # minimum lr
                    lr_ = opt.train.min_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

        # write last 10 mean dice
        last10_metric = last10_meter.get_metric()
        logging.warning('Mean dice in last 10 epochs: {:.4f}'.format(last10_metric['dice']))

        self.log_final()
        self.writer.close()

    """
    Noise-aware Training Stage.
    """
    def noise_aware_training(self):
        # init
        num_epochs = opt.train.n_epochs
        loss_meter, dice_meter = MultiLossMeter(), MultiLossMeter()
        loss_meter2, dice_meter2 = MultiLossMeter(), MultiLossMeter()
        if opt.framework.tri_net:
            loss_meter3, dice_meter3 = MultiLossMeter(), MultiLossMeter()

        loss_gap_epoch = MultiLossMeter()
        total_iter = {'train': 0, 'val': 0, 'test': 0}

        # warmup dataloader
        print('Warming up dataloader')
        print('Num per train_loader', sum(1 for _ in self.data_loaders['train']))
        print('Num per valid_loader', sum(1 for _ in self.data_loaders['val']))

        # mean dice of last 10 epochs
        last10_meter, last10_meter2 = MultiLossMeter(), MultiLossMeter()
        last10_meter.reset()
        last10_meter2.reset()

        # Set forget rate = 1 - selection_ratio/remember_rate with drop rate schedule.
        pfr_bg = np.ones(num_epochs + 1) * (1 - opt.framework.remember_rate_bg)
        pfr_bg[:opt.framework.warmup_epoch] = np.linspace(0, 1 - opt.framework.remember_rate_bg,
                                                          opt.framework.warmup_epoch)
        pfr_fg = np.ones(num_epochs + 1) * (1 - opt.framework.remember_rate_fg)
        pfr_fg[:opt.framework.warmup_epoch] = np.linspace(0, 1 - opt.framework.remember_rate_fg,
                                                          opt.framework.warmup_epoch)

        # Set correction ratio.
        if opt.framework.corr_ratio_fg < 0:
            opt.framework.corr_ratio_fg = round((1 - opt.framework.remember_rate_fg) * opt.framework.corr_ratio_decay,
                                                4)
        if opt.framework.corr_ratio_bg < 0:
            opt.framework.corr_ratio_bg = round((1 - opt.framework.remember_rate_bg) * opt.framework.corr_ratio_decay,
                                                4)
        epoch_notrain = 0
        update_epoch = 0

        first_E_tag = True
        not_improved_epochs = 0
        largest_gap_value = 0

        for epoch in tqdm(range(0, num_epochs + 1)):

            # Automatic switching to label refinement stage.
            if first_E_tag and opt.framework.Auto_E_step_epoch and (not_improved_epochs >= opt.framework.not_improved_epochs):

                # Save checkpoints for peer networks.
                snapshot = copy.deepcopy(self.model)
                snapshot2 = copy.deepcopy(self.model2)
                torch.save(snapshot.cpu().state_dict(), '{}/model_up_epoch_{}.pth'.format(opt.model_dir, epoch - 1))
                torch.save(snapshot2.cpu().state_dict(), '{}/model2_up_epoch_{}.pth'.format(opt.model_dir, epoch - 1))

                not_improved_epochs = 0
                largest_gap_value = 0
                update_epoch = epoch
                logging.warning('update_label')
                self.label_refinement(first_tag=first_E_tag)

                # Update forget rates and correct rates.
                pfr_bg = np.ones(num_epochs + 1) * (1 - opt.framework.remember_rate_bg)
                pfr_fg = np.ones(num_epochs + 1) * (1 - opt.framework.remember_rate_fg)
                opt.framework.corr_ratio_fg = round(
                    (1 - opt.framework.remember_rate_fg) * opt.framework.corr_ratio_decay, 4)
                opt.framework.corr_ratio_bg = round(
                    (1 - opt.framework.remember_rate_bg) * opt.framework.corr_ratio_decay, 4)

                first_E_tag = False

            # Alternate between training and validation.
            # The validation stage is used to evaluate models' performance on test set during training.
            # Note that this information is unknown for training models.
            # (We do not use it to choose hyper-parameters or do model selection).
            for phase in self.phases:
                if phase == 'train':
                    self.model.train(True)
                    self.model2.train(True)
                    if opt.framework.tri_net:
                        self.model3.train(True)
                else:
                    self.model.train(False)
                    self.model2.train(False)
                    if opt.framework.tri_net:
                        self.model3.train(False)
                    sum(1 for _ in self.data_loaders['val'])  # for debugging val_loader error: unexpected number
                loss_meter.reset(); dice_meter.reset();
                loss_meter2.reset(); dice_meter2.reset()
                if opt.framework.tri_net:
                    loss_meter3.reset();
                    dice_meter3.reset()
                loss_gap_epoch.reset();
                count_batch = 0
                while count_batch < opt.train.n_batches:
                    for i_batch, data in enumerate(self.data_loaders[phase]):
                        inputs = data['image'].float().cuda()
                        if not isinstance(data['gt'], list):
                            data['gt'] = [data['gt']]
                        targets = [it.long().cuda() for it in data['gt']]

                        outputs = self.model(inputs)
                        outputs2 = self.model2(inputs)

                        if opt.framework.tri_net:
                            outputs3 = self.model3(inputs)

                        if not isinstance(outputs, tuple):
                            outputs, outputs2 = tuple([outputs]), tuple([outputs2])
                            if opt.framework.tri_net:
                                outputs3 = tuple([outputs3])
                        preds = nn.Softmax(dim=1)(outputs[0] )[:, 1, ...]  # preds (B, H, W)
                        preds2 = nn.Softmax(dim=1)(outputs2[0])[:, 1, ...]
                        if opt.framework.tri_net:
                            preds3 = nn.Softmax(dim=1)(outputs3[0])[:, 1, ...]

                        # loss, backward
                        if phase == 'train' and epoch != epoch_notrain:
                            if not isinstance(data['clean_label'], list):
                                data['clean_label'] = [data['clean_label']]
                            if opt.data.load_superpixel_label:
                                superpixels = data['superpixel'].long().cuda()

                            # Compute pixel-wise loss map
                            losses, _ = self.criterion.forward_pixel(outputs[0], targets[0])
                            losses2, _ = self.criterion.forward_pixel(outputs2[0], targets[0])
                            if opt.framework.tri_net:
                                losses3, _ = self.criterion.forward_pixel(outputs3[0], targets[0])

                            # Apply selection during training, implementation including the proposed methods with
                            # superpixel-wise selection, and pixel-wise selection using peer networks/tri-networks.
                            if opt.framework.pixel_select:

                                # The proposed: superpixel-wise sample selection.
                                if opt.data.load_superpixel_label and opt.framework.superpixel_select:

                                    with torch.no_grad():

                                        # Calculate superpixe-wise probability.
                                        preds_sm = self.super_pixel_smoothing(preds, superpixels)
                                        probs_sm = torch.zeros(outputs[0].shape).cuda()
                                        probs_sm[:, 1] = preds_sm
                                        probs_sm[:, 0] = 1 - preds_sm
                                        preds_sm2 = self.super_pixel_smoothing(preds2, superpixels)
                                        probs_sm2 = torch.zeros(outputs2[0].shape).cuda()
                                        probs_sm2[:, 1] = preds_sm2
                                        probs_sm2[:, 0] = 1 - preds_sm2

                                        # Calculate superpixel-wise loss.
                                        losses_, _ = self.criterion.forward_pixel(probs_sm, targets[0], x_style='prob')
                                        losses2_, _ = self.criterion.forward_pixel(probs_sm2, targets[0],
                                                                                   x_style='prob')
                                        # Select superpixels with small losses.
                                        pixel_select_1, pixel_select_2 = \
                                            self.pixel_loss_selection(losses_, losses2_, [probs_sm], [probs_sm2], targets,
                                                                      (pfr_bg[epoch], pfr_fg[epoch]), kl_style='prob')

                                # Pixel-wise sample selectin with Tri-network.
                                elif opt.framework.tri_net:
                                    pixel_select_1, pixel_select_2, pixel_select_3 = \
                                        self.tri_net_selection(losses, losses2, losses3, targets,
                                                               ( pfr_bg[epoch], pfr_fg[epoch]))

                                # Pixel-wise sample selection with peer networks.
                                else:
                                    pixel_select_1, pixel_select_2 = \
                                        self.pixel_loss_selection(losses, losses2, outputs, outputs2, targets,
                                                                  (pfr_bg[epoch], pfr_fg[epoch]))
                                # compute loss with selection map
                                select_1 = pixel_select_1
                                select_2 = pixel_select_2
                                if opt.framework.tri_net:
                                    select_3 = pixel_select_3
                            else:
                                # Do not perform sample selection, use all data for network updating.
                                select_1 = torch.ones_like(targets[0], dtype=bool).flatten()
                                select_2 = torch.ones_like(targets[0], dtype=bool).flatten()

                            if opt.framework.JoCoR:
                                kl = self.kl_loss_compute(outputs[0], outputs2[0], reduce=False)
                                kl2 = self.kl_loss_compute(outputs2[0], outputs[0], reduce=False)
                                losses[0] = (1 - opt.framework.co_lambda) * (
                                            losses[0] + losses2[0]) + opt.framework.co_lambda * (kl + kl2)

                            loss = losses[0].flatten()
                            loss2 = losses2[0].flatten()
                            if opt.framework.tri_net:
                                loss3 = losses3[0].flatten()

                            target = targets[0].flatten()

                            if opt.framework.tri_net:
                                losses, loss_names = self.criterion.forward_select(loss[select_1], target[select_1])
                                losses2, loss_names2 = self.criterion.forward_select(loss2[select_2], target[select_2])
                                losses3, loss_names3 = self.criterion.forward_select(loss3[select_3], target[select_3])
                            else:
                                losses, loss_names = self.criterion.forward_select(loss[select_2], target[select_2])
                                losses2, loss_names2 = self.criterion.forward_select(loss2[select_1], target[select_1])

                            # Compute the loss gap between selected data and unselected data.
                            # We use loss gap as a stopping criterion.
                            loss_noisy, _ = self.criterion.forward_select(loss[~select_2], target[~select_2])
                            loss_clean, _ = self.criterion.forward_select(loss[select_2], target[select_2])
                            loss_gap = loss_noisy[0] - loss_clean[0]
                            if torch.isnan(loss_gap):
                                loss_gap = - loss_clean[0]
                            loss_gap_epoch.update([loss_gap], ['loss_gap'])

                            # Backward
                            loss = losses[0]
                            loss2 = losses2[0]
                            self.optimizer.zero_grad()
                            if opt.framework.JoCoR:
                                loss.backward()
                            elif opt.framework.tri_net:
                                (loss + loss2 + losses3[0]).backward()
                            else:
                                (loss + loss2).backward()
                            self.optimizer.step()

                        else:
                            with torch.no_grad():
                                losses, loss_names = self.criterion.forward(outputs[0], targets[0])
                                losses2, loss_names2 = self.criterion.forward(outputs2[0], targets[0])
                                if opt.framework.tri_net:
                                    losses3, loss_names3 = self.criterion.forward(outputs3[0], targets[0])
                                    loss3 = losses3[0]

                        # loss meter, dice meter
                        loss_meter.update(losses, loss_names)
                        loss_meter2.update(losses2, loss_names2)
                        if opt.framework.tri_net:
                            loss_meter3.update(losses3, loss_names3)
                            dices3, dice_names3 = self.metric.forward(preds3, targets[0])
                            dice_meter3.update(dices3, dice_names3)

                        dices, dice_names = self.metric.forward(preds, targets[0])
                        dices2, dice_names2 = self.metric.forward(preds2, targets[0])
                        dice_meter.update(dices, dice_names)
                        dice_meter2.update(dices2, dice_names2)

                        count_batch += 1
                        if count_batch >= opt.train.n_batches:
                            break

                        # release memory
                        del inputs, targets, preds, preds2, outputs, outputs2, losses, losses2, dices, dices2,
                        gc.collect()

                    if phase == 'val':
                        break
                    if phase == 'train' and epoch == epoch_notrain:
                        break

                # epoch metric
                avg_terms = loss_meter.get_metric()
                avg_terms2 = loss_meter2.get_metric()
                dice_terms = dice_meter.get_metric()
                dice_terms2 = dice_meter2.get_metric()

                if opt.framework.tri_net:
                    avg_terms3 = loss_meter3.get_metric()
                    dice_terms3 = dice_meter3.get_metric()
                if phase == 'train' and epoch != epoch_notrain:
                    gap_term = loss_gap_epoch.get_metric()
                    gap_value = gap_term['loss_gap']

                    if gap_value > largest_gap_value:
                        not_improved_epochs = 0
                        largest_gap_value = gap_value
                    else:
                        not_improved_epochs += 1

                    avg_terms = {**avg_terms, **dice_terms, **gap_term}
                    avg_terms2 = {**avg_terms2, **dice_terms2}

                else:
                    avg_terms = {**avg_terms, **dice_terms}
                    avg_terms2 = {**avg_terms2, **dice_terms2}

                if opt.framework.tri_net:
                    avg_terms3 = {**avg_terms3, **dice_terms3}

                self.plot_curves_multi(avg_terms, epoch, phase=phase)
                self.plot_curves_multi(avg_terms2, epoch, phase=phase, co_teaching=True)

                if opt.framework.tri_net:
                    self.plot_curves_multi(avg_terms3, epoch, phase=phase, tri_net=True)

                if phase == 'val':
                    ## update last 10 dice meter
                    if epoch > num_epochs - 10:
                        last10_meter.update([avg_terms['dice']], ['dice'])
                        last10_meter2.update([avg_terms2['dice']], ['dice'])

                self.log_epoch(avg_terms, epoch, phase)
                self.log_epoch(avg_terms2, epoch, phase, duplicate=True)

                if opt.framework.tri_net:
                    self.log_epoch(avg_terms3, epoch, phase, duplicate=True)

            self.log_time()


            # save snapshot
            if epoch == 200:
                snapshot = copy.deepcopy(self.model)
                snapshot2 = copy.deepcopy(self.model2)
                torch.save(snapshot.cpu().state_dict(), '{}/model_epoch_{}.pth'.format(opt.model_dir, epoch))
                torch.save(snapshot2.cpu().state_dict(), '{}/model2_epoch_{}.pth'.format(opt.model_dir, epoch))

            # scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_terms['dice'])
                else:
                    self.scheduler.step()
            elif opt.train.lr_decay == 'poly':
                lr_ = opt.train.lr * (1 - epoch / opt.train.n_epochs) ** 0.9  # power=0.9
                if lr_ < opt.train.min_lr:  # minimum lr
                    lr_ = opt.train.min_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_
            elif opt.train.lr_decay == 'constant_decay' and epoch == opt.framework.E_step_epoch - 1:
                lr_ = opt.train.lr / 2
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

        # write last 10 mean dice
        last10_metric = last10_meter.get_metric()
        last10_metric2 = last10_meter2.get_metric()
        logging.warning('Mean dice in last 10 epochs: Model1 {:.4f} Model2 {:.4f} '.format(last10_metric['dice'],
                                                                                           last10_metric2['dice']))
        logging.warning('Update label at Epoch {} '.format(update_epoch))
        self.log_final()
        self.writer.close()

    """
    Label Refinement Stage.
    """
    def label_refinement(self, first_tag=False, iterate_order=0):

        def save_updated_label(save_path, mask):
            label = {opt.data.dataname: mask}
            with open(save_path, 'w') as f:
                json.dump(label, f, cls=NpEncoder)

        def label_update(probs, targets, bg_co_rate, fg_co_rate, loss=None, correct_operation=True):
            if iterate_order > 0:
                bg_co_rate = bg_co_rate / (4 ** iterate_order)
                fg_co_rate = fg_co_rate / (4 ** iterate_order)

            noisy_labels = targets[0][:, 0]
            bg_co_num = int(bg_co_rate * len(probs[noisy_labels == 0]))
            fg_co_num = int(fg_co_rate * len(probs[noisy_labels == 1]))

            if loss is None:
                # label from given bg to corrected fg
                value_fg, _ = torch.topk(probs[noisy_labels == 0], bg_co_num, largest=True, sorted=True)
                # label from given fg to corrected bg
                value_bg, _ = torch.topk(probs[noisy_labels == 1], fg_co_num, largest=False, sorted=True)
            else:
                value_fg, _ = torch.topk(loss[noisy_labels == 0], bg_co_num, largest=True, sorted=True)
                value_bg, _ = torch.topk(loss[noisy_labels == 1], fg_co_num, largest=True, sorted=True)

            thresh_fg, thresh_bg = value_fg[-1], value_bg[-1]

            # external prob limit
            if opt.framework.conf_thres_fg is not None:
                if opt.framework.conf_thres_fg > thresh_fg:
                    thresh_fg = opt.framework.conf_thres_fg
                if opt.framework.conf_thres_bg < thresh_bg:
                    thresh_bg = opt.framework.conf_thres_bg

            if loss is None:
                correct_fg = (probs > thresh_fg) & (noisy_labels == 0)
                correct_bg = (probs < thresh_bg) & (noisy_labels == 1)
            else:
                correct_fg = (loss > thresh_fg) & (noisy_labels == 0)
                correct_bg = (loss > thresh_bg) & (noisy_labels == 1)

            if correct_operation:
                # modify noisy label
                pseudo = noisy_labels.clone()
                pseudo[correct_fg == 1] = (probs[correct_fg == 1] > 0.5).long()  # 1
                pseudo[correct_bg == 1] = (probs[correct_bg == 1] > 0.5).long()  # 0

                return pseudo
            else:
                return correct_fg, correct_bg

        def update_remember_rate_linear():
            '''
            Approach:
                MAX(R, MIN(R*1.1,threshold)
            '''
            rem_rate = {}
            rem_thres_fg = opt.framework.rem_thres
            if opt.framework.rem_thres_bg < 1:
                rem_thres_bg = opt.framework.rem_thres_bg
            else:
                rem_thres_bg = opt.framework.rem_thres

            rem_rate['fg'] = max(opt.framework.remember_rate_fg,
                                 min(opt.framework.remember_rate_fg * opt.framework.rem_linear_param, rem_thres_fg))
            rem_rate['bg'] = max(opt.framework.remember_rate_bg,
                                 min(opt.framework.remember_rate_bg * opt.framework.rem_linear_param, rem_thres_bg))
            return rem_rate


        # inference train set, save to a fix dir, for later dataloader.
        # 0. Params for uncertainty filter
        start_time = datetime.datetime.now()
        logging.warning('E step into: generated new pseudo for %s.' % args.id)
        save_em_pseudo_dir = opt.data.em_save_pseudo_dir + '/%s/' % args.id
        if first_tag and os.path.exists(save_em_pseudo_dir):  # clear save_em_pseudo_dir
            shutil.rmtree(save_em_pseudo_dir)
        if not os.path.exists(save_em_pseudo_dir):
            os.makedirs(save_em_pseudo_dir)

        self.model.train(False)
        self.model2.train(False)

        print('Warming up E_train_loader')
        print('Num per E_train_loader', sum(1 for _ in self.E_train_loader))

        # 1. dataloader, model forward
        for i_batch, data in enumerate(self.E_train_loader):
            inputs = data['image'].float().cuda()
            if not isinstance(data['gt'], list):
                data['gt'] = [data['gt']]
            targets = [it.long().cuda() for it in data['gt']]

            # superpixel related: correction
            if opt.data.load_superpixel_label:
                superpixels = data['superpixel'].long().cuda()

            with torch.no_grad():
                outputs = self.model(inputs)
                probs = nn.Softmax(dim=1)(outputs)[:, 1, ...]
                outputs2 = self.model2(inputs)
                probs2 = nn.Softmax(dim=1)(outputs2)[:, 1, ...]
                # # average probability
                probs = (probs + probs2) / 2

            # Superpixe-wise label refinement
            if opt.data.load_superpixel_label and opt.framework.sp_label_update_style:
                if opt.framework.sp_label_update_style == 'mean':
                    probs_ = self.super_pixel_smoothing(probs, superpixels)
                    pseudos = label_update(probs_, targets, opt.framework.corr_ratio_bg, opt.framework.corr_ratio_fg)

            # Pixel-wise label refinement
            else:
                pseudos = label_update(probs, targets, opt.framework.corr_ratio_bg, opt.framework.corr_ratio_fg)

            batch_size = pseudos.shape[0]
            for kth in range(batch_size):
                case_id = self.save_val_ids['train'][i_batch * batch_size + kth]
                save_path = save_em_pseudo_dir + case_id + '.json'
                pseudo = pseudos[kth].cpu().numpy().astype(np.uint8)
                save_updated_label(save_path, pseudo)

        logging.warning('E step done: generated new pseudo labels for M step.')


        if opt.framework.E_update_remember_rate_linear:
            remember_rates = update_remember_rate_linear()
            opt.framework.remember_rate_fg, opt.framework.remember_rate_bg = remember_rates['fg'], remember_rates['bg']
            logging.warning('E step update remember rate: FG %.4f, BG %.4f.' % (
                opt.framework.remember_rate_fg, opt.framework.remember_rate_bg))

        print('E step time cost', datetime.datetime.now() - start_time)

    """
    We use the prediction of network 1 for inference.
    """
    def inference(self):
        # init
        for phase in ['train', 'val', 'test']:
            if phase in self.phases:
                print('Num per %s_loader' % phase, sum(1 for _ in self.data_loaders[phase]))
        loss_meter, dice_meter = MultiLossMeter(), MultiLossMeter()
        self.model.train(False)

        for phase in self.phases:
            save_dir = os.path.join(os.getcwd(), 'output/%s_%s' % (args.id, phase))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            log_path = os.path.join(save_dir, 'log.txt')
            log = open(log_path, 'w')
            log.write('Dice metrics\n')
            dice_meter.reset()
            dice_list = [];
            probs = []

            for i_batch, data in tqdm(enumerate(self.data_loaders[phase])):
                # model forward
                ## co-teaching data keys for train set: 'gt': noisy label, 'clean_label': real clean mask
                inputs = data['image'].float().cuda()
                if not isinstance(data['gt'], list):
                    data['gt'] = [data['gt']]
                    if 'clean_label' in data.keys():
                        data['clean_label'] = [data['clean_label']]
                targets = [it.long().cuda() for it in data['gt']]

                with torch.no_grad():
                    outputs = self.model(inputs)
                    if not isinstance(outputs, tuple):
                        outputs = tuple([outputs])
                    preds = nn.Softmax(dim=1)(outputs[0])[:, 1, ...]
                    losses, _ = self.criterion.forward_pixel(outputs[0], targets[0])

                # dice meter
                if phase != 'test':
                    gt_key = 'gt' if 'clean_label' not in data.keys() else 'clean_label'
                    gt = [it.long().cuda() for it in data[gt_key]]
                    dices, dice_names = self.metric.forward(preds.cpu(), gt[0].cpu())
                else:
                    dices = [torch.tensor(0.00, requires_grad=True)];
                    dice_names = ['dice']
                dice_meter.update(dices, dice_names)
                dice_list.append(dices[0])

                _id = self.val_ids[phase][i_batch] if self.val_ids is not None else i_batch

                # save preds
                if args.save_preds:
                    gt = gt[0][:, 0, ...]
                    self.save_pred([(preds > 0.5).detach().cpu(), gt.int().cpu()],
                                   names=['pred', 'gt_label'],
                                   id=_id, title=args.id, phase=phase, save_type='2d')

                log.write('%s: dice_fg %.4f\n' % (_id, dices[0]))

            # avg metric
            avg_terms = dice_meter.get_metric()
            log.write('AVG: dice_fg %.4f\n' % (avg_terms['dice']))
            log.write('Samples std: %.4f\n' % torch.std(torch.tensor(dice_list)).item())
            log.close()
            # print metrics
            print('METRICS:')
            for key, value in avg_terms.items():
                print(key, value)
            print()

    """
    We use superpixels as a guidance during the iterative leaning.
    """
    def super_pixel_smoothing(self, pixel_map, superpixels, operation='mean', alpha=0.5):
        '''
        :param pixel_map:   shape (B, 2, H, W) or BHW
        :param superpixels:  shape (B,H, W) or (B,1,H,W)
        :param operation: mean, mode
        :return: pixel_map_sm:   shape (B, H, W)
        '''

        eps = 1e-6
        B, H, W = superpixels[:, 0].shape
        pixel_map = pixel_map.reshape(B, H, W)
        pixel_map_list = []

        for ith in range(B):
            pixel_map_, superpixel = pixel_map[ith], superpixels[ith, 0]  # (H, W)
            onehot_mask = F.one_hot(superpixel)  # (H, W, K)
            K = onehot_mask.shape[2]
            pixel_map_fla = pixel_map_.unsqueeze(2).repeat(1, 1, K)  # (H, W, K)
            pixel_map_onehot = pixel_map_fla * onehot_mask  # (H, W, K)

            # Option 1: Use mean
            if operation == 'mean':
                smooth_value = pixel_map_onehot.sum(dim=(0, 1)) / ((onehot_mask == 1).sum(dim=(0, 1)) + eps)  # (K)

            # Option 2: Use mode:
            if operation == 'mode':
                smooth_value = (pixel_map_onehot.sum(dim=(0, 1)) > alpha * (onehot_mask == 1).sum(
                    dim=(0, 1))).float()  # (K)

            pixel_map_sm_ = torch.matmul(onehot_mask.float(), smooth_value.float())  # (H, W)
            pixel_map_list.append(pixel_map_sm_.unsqueeze(0))

        pixel_map_sm = torch.cat(pixel_map_list, axis=0)

        return pixel_map_sm

    """
    The pixel-wise selection process, contain co-teaching and JoCoR strategies.
    """
    def pixel_loss_selection(self, losses, losses2, outputs, outputs2, targets, forget_rate, kl_style='logit'):

        # Prepare variables to be used.
        select_1 = torch.zeros_like(targets[0], dtype=bool).flatten()
        select_2 = torch.zeros_like(targets[0], dtype=bool).flatten()
        output, output2 = outputs[0], outputs2[0]
        criterion, criterion2 = losses[0], losses2[0]

        # Choose which criterion to use, loss, or joint loss.
        if opt.framework.JoCoR:
            kl = self.kl_loss_compute(output, output2, reduce=False, kl_style=kl_style)
            kl2 = self.kl_loss_compute(output2, output, reduce=False, kl_style=kl_style)
            criterion = (1 - opt.framework.co_lambda) * (criterion + criterion2) + opt.framework.co_lambda * (kl + kl2)
            criterion2 = criterion

        criterion, criterion2 = criterion.flatten(), criterion2.flatten()

        target = targets[0].flatten()
        target2 = targets[0].flatten()

        # Select partial data with small losses.
        # Background pixels and foreground pixels are selected separately.
        criterion_bg = criterion[target == 0]
        criterion_fg = criterion[target == 1]
        criterion_bg2 = criterion2[target2 == 0]
        criterion_fg2 = criterion2[target2 == 1]

        num_bg = len(criterion[targets[0].flatten() == 0])
        num_fg = len(criterion[targets[0].flatten() == 1])
        value_bg, _ = torch.topk(criterion_bg, int((forget_rate[0]) * num_bg), largest=True, sorted=True)
        value_fg, _ = torch.topk(criterion_fg, int((forget_rate[1]) * num_fg), largest=True, sorted=True)
        thresh_bg, thresh_fg = value_bg[-1], value_fg[-1]

        # JoCoR only has one joint loss.
        if not opt.framework.JoCoR:
            value_bg2, _ = torch.topk(criterion_bg2, int((forget_rate[0]) * num_bg), largest=True, sorted=True)
            value_fg2, _ = torch.topk(criterion_fg2, int((forget_rate[1]) * num_fg), largest=True, sorted=True)
            thresh_bg2, thresh_fg2 = value_bg2[-1], value_fg2[-1]

        # Generate selection map.
        select_1[(target == 0) & (criterion < thresh_bg)] = True  # selected bg elements
        select_1[(target == 1) & (criterion < thresh_fg)] = True  # selected fg elements
        if not opt.framework.JoCoR:
            select_2[(target2 == 0) & (criterion2 < thresh_bg2)] = True  # selected bg elements
            select_2[(target2 == 1) & (criterion2 < thresh_fg2)] = True  # selected fg elements

        if opt.framework.JoCoR:
            return select_1, select_1
        else:
            return select_1, select_2

    def tri_net_selection(self, losses, losses2, losses3, targets, forget_rate):
        criterion = (losses2[0].flatten() - losses3[0].flatten()).abs()
        criterion2 = (losses[0].flatten() - losses3[0].flatten()).abs()
        criterion3 = (losses[0].flatten() - losses2[0].flatten()).abs()

        target = targets[0].flatten()

        criterion_bg = criterion[target == 0]
        criterion_fg = criterion[target == 1]
        criterion_bg2 = criterion2[target == 0]
        criterion_fg2 = criterion2[target == 1]
        criterion_bg3 = criterion3[target == 0]
        criterion_fg3 = criterion3[target == 1]

        num_bg = len(criterion[target == 0])
        num_fg = len(criterion[target == 1])

        value_bg, _ = torch.topk(criterion_bg, int((forget_rate[0]) * num_bg), largest=True, sorted=True)
        value_fg, _ = torch.topk(criterion_fg, int((forget_rate[1]) * num_fg), largest=True, sorted=True)
        thresh_bg, thresh_fg = value_bg[-1], value_fg[-1]

        value_bg2, _ = torch.topk(criterion_bg2, int((forget_rate[0]) * num_bg), largest=True, sorted=True)
        value_fg2, _ = torch.topk(criterion_fg2, int((forget_rate[1]) * num_fg), largest=True, sorted=True)
        thresh_bg2, thresh_fg2 = value_bg2[-1], value_fg2[-1]

        value_bg3, _ = torch.topk(criterion_bg3, int((forget_rate[0]) * num_bg), largest=True, sorted=True)
        value_fg3, _ = torch.topk(criterion_fg3, int((forget_rate[1]) * num_fg), largest=True, sorted=True)
        thresh_bg3, thresh_fg3 = value_bg3[-1], value_fg3[-1]

        select_1 = torch.zeros_like(targets[0], dtype=bool).flatten()
        select_2 = torch.zeros_like(targets[0], dtype=bool).flatten()
        select_3 = torch.zeros_like(targets[0], dtype=bool).flatten()

        select_1[(target == 0) & (criterion < thresh_bg)] = True
        select_1[(target == 1) & (criterion < thresh_fg)] = True
        select_2[(target == 0) & (criterion2 < thresh_bg2)] = True
        select_2[(target == 1) & (criterion2 < thresh_fg2)] = True
        select_3[(target == 0) & (criterion3 < thresh_bg3)] = True
        select_3[(target == 1) & (criterion3 < thresh_fg3)] = True

        return select_1, select_2, select_3

    def kl_loss_compute(self, pred, soft_targets, reduce=True, kl_style='logit'):
        if kl_style == 'logit':
            kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduction='none')
        else:
            kl = F.kl_div(torch.log(pred), soft_targets, reduction='none')
        if reduce:
            return torch.mean(torch.sum(kl, dim=1))
        else:
            return torch.sum(kl, 1)

    def log_init(self):

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        logfile = opt.log_dir + "/log_{}.txt".format(args.id)
        fh = logging.FileHandler(logfile)  # , mode='w') # whether to clean previous file
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        logging.info(str(opt))
        logging.info('Time: %s' % datetime.datetime.now())
        # print(opt)

    def log_batch(self, data, epoch, i_batch):
        loss, dice = data['loss'], data['dice']
        phrase = 'Epoch: {:4.0f} i_batch: {:4.0f} mDice: {:.6f} Loss: {:.6f} '.format(
            epoch, i_batch, dice, loss)
        if args.demo == '':
            logging.info(phrase)
        else:
            logging.warning(phrase)

    def log_epoch(self, data, epoch, phase, duplicate=False):
        # print(duplicate)
        # import ipdb
        # ipdb.set_trace()
        str = '{} Epoch: {} '.format(phase, epoch)
        for key, value in data.items():
            str = str + '{}: {:.6f} '.format(key, value)
        logging.warning(str)

    def log_time(self):
        time_elapsed = time.time() - self.since
        logging.warning('Time till now {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
        logging.warning('Time: %s' % datetime.datetime.now())

    def log_final(self):
        time_elapsed = time.time() - self.since
        logging.warning('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))

    def plot_curves_multi(self, data, epoch, iter=-1, phase='train', co_teaching=False, tri_net=False):
        ''' data: multiple loss terms '''
        group_name = 'epoch_verbose' if iter < 0 else 'iter_verbose'
        count = epoch if iter < 0 else iter
        if co_teaching:
            group_name = group_name + "2"
        if tri_net:
            group_name = group_name + "3"
        for key, value in data.items():
            self.writer.add_scalar('%s/%s_%s' % (group_name, phase, key), value, count)





