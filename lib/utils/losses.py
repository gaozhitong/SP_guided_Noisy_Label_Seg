import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None,
                 use_ce_loss=True):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        :param use_ce_loss: no channel dim in CELoss
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

        self.use_ce_loss = use_ce_loss

    def compute_loss(self, x, y, pixel_weights=None, disable_auto_weight=False, per_sample_cal=False):
        '''
        Note that binary-class setting and per-sample calculation by default.
        :param pixel_weights: pixel-level weight map
        :param disable_auto_weight: default we use automatic class weights
        :param per_sample_cal: if True, calculate class weights, loss for each sample of a batch
        :return:
        '''
        eps = 1e-6
        max_fg_weight = 1e6

        # 1. auto class weights
        if (not disable_auto_weight):
            # A. calculate loss per sample: feasible for 3D volume and highly imbalanced case with large variance
            if per_sample_cal:
                loss = []
                for b in range(x.shape[0]):

                    fg_weight = ((y[b:b+1] == 0).sum().float() / ((y[b:b+1] == 1).sum().float() + eps)).item()
                    if fg_weight > max_fg_weight:
                        fg_weight = max_fg_weight
                    class_weights = torch.Tensor([1.0, fg_weight]).cuda() # only for weak now

                    if pixel_weights is not None:
                        loss.append(self.loss(x[b:b+1], y[b:b+1], pixel_weights=pixel_weights[b:b + 1], weight=class_weights))
                    else:
                        loss.append(self.loss(x[b:b+1], y[b:b+1], weight=class_weights))

                return sum(loss) / len(loss)
            # B. calculate loss per batch: feasible for 2D images
            else:
                fg_weight = ((y == 0).sum().float() / ((y == 1).sum().float() + eps)).item()
                if fg_weight > max_fg_weight:
                    fg_weight = max_fg_weight
                class_weights = torch.Tensor([1.0, fg_weight]).cuda()   # only for binary-class now
                if pixel_weights is not None:
                    return self.loss(x, y, pixel_weights=pixel_weights, weight=class_weights) 
                else:
                    return self.loss(x, y, weight=class_weights)        

        # 2. no class weights
        else:
            return self.loss(x, y)

    def forward(self, x, y, pixel_weights=None, disable_auto_weight=False):
        '''
        :param pixel_weights: pixel-level weight map
        :param disable_auto_weight: default we use automatic class weights
        :return:
        '''
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        # shuailin add
        losses = []; loss_names = []
        if self.use_ce_loss:
            y = [it[:, 0, ...] for it in y]     # no channel dim in CELoss

        # l = weights[0] * self.loss(x[0], y[0])
        l = weights[0] * self.compute_loss(x[0], y[0], pixel_weights=pixel_weights, disable_auto_weight=disable_auto_weight)
        losses.append(l); loss_names.append('d0')
        for i in range(1, len(x)):
            if weights[i] != 0:
                # loss = weights[i] * self.loss(x[i], y[i])
                loss = weights[i] * self.compute_loss(x[i], y[i], pixel_weights=pixel_weights, disable_auto_weight=disable_auto_weight)
                if l.shape == loss.shape:
                    l = l + loss
                losses.append(loss); loss_names.append('d%d' % i)

        # return l
        return [l] + losses, ['loss'] + loss_names

class CELoss2(nn.Module):
    '''
    - Optional mode: reduction = 'mean' or 'none'
    - Automantic weight, no need to provide weight
    - No deep supervision scales
    '''
    def __init__(self):
        super(CELoss2, self).__init__()

    def compute_class_weights(self, y, eps=1e-6, max_fg_weight=1e6):
        '''
        :return: foreground/background weights, only for binary-class now
        '''
        fg_weight = ((y == 0).sum().float() / ((y == 1).sum().float() + eps)).item()
        if fg_weight > max_fg_weight:
            fg_weight = max_fg_weight
        elif fg_weight < 1:
            fg_weight = 1
        class_weights = torch.Tensor([1.0, fg_weight]).cuda()
        return class_weights

    def reduce_loss(self, loss, y, class_weights=None):
        if class_weights is None:
            return loss.mean()

        fg_sum, bg_sum = (y == 1).long().sum(), (y == 0).long().sum()
        fg_weight, bg_weight = class_weights[1], class_weights[0]
        loss = (fg_weight * loss[y == 1].sum() + bg_weight * loss[y == 0].sum()) / (
                fg_sum * fg_weight + bg_sum * bg_weight)
        return loss

    def forward(self, x, y, pixel_weights=None, ignore_index=255):
        '''
        :return: loss scalar, class weights used, pixel_weights accepted
        '''
        pass
        y = y[:, 0, ...]  # no channel dim in CELoss

        # automatic class weights
        class_weights = self.compute_class_weights(y)

        # default
        if pixel_weights is None:
            loss_func = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
            ce_loss = loss_func.forward(x, y)

        # given pixel weights
        else:
            loss_func = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
            ce_loss = loss_func.forward(x, y)
            ce_loss = torch.mul(ce_loss, pixel_weights)
            ce_loss = self.reduce_loss(ce_loss, y, class_weights)

        return [ce_loss], ['loss']

    def forward_pixel(self, x, y, x_style='logit'):
        '''
        x_style:  {'logit', 'prob}
        return pixel-wise loss map, no class weights used
        '''

        y = y[:, 0, ...]  # no channel dim in CELoss

        if x_style == 'logit':    # default
            loss_func = nn.CrossEntropyLoss(reduction='none')
            ce_loss = loss_func.forward(x, y)

        elif x_style == 'prob':  # x is probability
            ce_loss = F.nll_loss(torch.log(x), y, reduction='none')

        return [ce_loss], ['loss']


    def forward_select(self, loss, y, pixel_weights=None):
        '''
        x: selected pixel-wise loss
        y: selected label map
        Return class weighted loss scalar
        '''
        # automatic class weights
        class_weights = self.compute_class_weights(y)
        if pixel_weights is not None:
            loss = torch.mul(loss, pixel_weights)
        ce_loss = self.reduce_loss(loss, y, class_weights)
        return [ce_loss], ['loss']



class CELoss(nn.CrossEntropyLoss):
    def __init__(self, weight=torch.Tensor([1.0, 1.0]).cuda(), pixel_weights=None, none_reduction=False):
        self.given_weight = weight
        self.pixel_weights = pixel_weights
        if pixel_weights is not None or none_reduction:
            super().__init__(reduction='none')
        else:
            super().__init__(weight=weight)

    def forward(self, input, target, weight=None, pixel_weights=None):

        if weight is not None:
            self.weight = weight
        if pixel_weights is not None:
            self.pixel_weights = pixel_weights

        ce_loss = super().forward(input, target)

        if self.pixel_weights is not None:
            ce_loss = torch.mul(ce_loss, pixel_weights)
            # ce_loss = ce_loss[target != self.ignore_index].mean()
            # hand-written weighted
            fg_sum, bg_sum = (target==1).long().sum(), (target==0).long().sum()
            fg_weight, bg_weight = self.given_weight[1], self.given_weight[0]
            ce_loss = (fg_weight * ce_loss[target==1].sum() + bg_weight * ce_loss[target==0].sum()) / (fg_sum*fg_weight + bg_sum*bg_weight)

        return ce_loss

class DiceLoss(object):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, score, target, smooth=1e-8):
        target = target.float()

        # filter ignored region
        if self.ignore_index is not None:
            score = score[target != self.ignore_index]
            target = target[target != self.ignore_index]

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss

        return [loss], ['dice_loss']

class CEDiceLoss(object):
    def __init__(self, weight=torch.Tensor([1.0, 1.0]).cuda(), ignore_index=255, multi_loss=True):
        self.multi_loss = multi_loss
        self.ignore_index = ignore_index
        self.celoss = CELoss(weight, ignore_index)
        self.dice_loss = DiceLoss()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, input, target, multi_loss=False):
        pred = nn.Softmax(dim=1)(input)[:, 1, ...]
        if self.multi_loss:  #self.multi_loss:
            ce_losses, ce_losses_names = self.celoss(input, target)
            dice_loss, dice_name = self.dice_loss(pred, target)
            total_loss, total_name = ce_losses[0] + dice_loss[0], 'total_loss'
            return [total_loss]+ce_losses+dice_loss, [total_name]+ce_losses_names+dice_name
        else:
            ce_loss = self.celoss(input, target, multi_loss)
            dice_loss, dice_name = self.dice_loss(pred, target)
            total_loss = ce_loss + dice_loss[0]
            return total_loss


########################## Metrics #####################

class DiceMetric(object):
    '''
    Input: preds: predicted probability
    '''
    def __init__(self, dice_each_class=False, smooth=1e-8):
        self.dice_each_class = dice_each_class
        self.smooth = smooth

    def forward(self, preds, gts):
        # preds, gts = preds.detach(), gts.detach()

        # fg dice
        dice = 0
        batch = preds.shape[0]
        for ith in range(batch):
            dice += self.dice_func(preds[ith], gts[ith])
        dice /= batch
        dice_fg = dice

        # bg dice
        if self.dice_each_class:
            dice = 0
            batch = preds.shape[0]
            for ith in range(batch):
                dice += self.dice_func(preds[ith], gts[ith], type='bg')
            dice /= batch
            dice_bg = dice

            return [dice_fg, dice_bg], ['dice', 'dice_bg']
        return [dice_fg], ['dice']


    def dice_func(self, pred, gt, type='fg'):
        if type == 'fg':
            pred = pred > 0.5
            label = gt > 0
        else:
            pred = pred < 0.5
            label = gt == 0
        inter_size = torch.sum(((pred * label) > 0).float())
        sum_size = (torch.sum(pred) + torch.sum(label)).float()
        dice = (2 * inter_size + self.smooth) / (sum_size + self.smooth)
        return dice

