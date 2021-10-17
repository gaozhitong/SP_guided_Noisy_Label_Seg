'''
Calculate the undersegmentation error of superpixels compare to GT.
'''

import json
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm
import ipdb


def ue_func(mask, gt_mask):
    over_segment_area = 0
    K = np.unique(mask) # numbers of superpixels
    for i in K:
        if ((mask == i) & (gt_mask == 1)).any():
            over_segment_area += np.sum(mask==i)
    gt_area = np.sum(gt_mask == 1)
    ue = (over_segment_area - gt_area) * 1.0 / gt_area
    return ue

def dice_func(pred, gt, type='fg'):
    smooth = 1e-8
    if type == 'fg':
        pred = pred > 0.5
        label = gt > 0
    else:
        pred = pred < 0.5
        label = gt == 0
    inter_size = np.sum(((pred * label) > 0))
    sum_size = (np.sum(pred) + np.sum(label))
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice


def dice_func(mask, gt_mask):
    smooth = 1e-8
    segment_mask = np.zeros_like(gt_mask)
    K = np.unique(mask) # numbers of superpixels
    for i in K:
        if np.sum((mask == i) & (gt_mask == 1)) >= 1/2 * np.sum(mask == i):
            segment_mask[(mask == i)] = 1
    inter_size = np.sum(((segment_mask * gt_mask) > 0))
    sum_size = (np.sum(segment_mask) + np.sum(gt_mask))
    dice = (2 * inter_size + smooth) / (sum_size + smooth)
    return dice

def stat():
    stat = {'ue': [], 'dice':[]}
    super_paths = sorted(glob(Params['super_dir'] + '*.json'))
    gt_paths = sorted(glob(Params['clean_dir'] + '*.json'))
    for ith in tqdm(range(len(gt_paths))):
        super_path, gt_path = super_paths[ith], gt_paths[ith]
        with open(super_path, 'rb') as f:
            mask = json.load(f)
        mask = np.array(mask)
        with open(gt_path, 'rb') as f:
            gt_mask = json.load(f)[Params['class']]
        gt_mask = np.array(gt_mask)

        stat['ue'].append(ue_func(mask, gt_mask))
        stat['dice'].append(dice_func(mask, gt_mask))

    mean_ue = np.array(stat['ue']).mean()
    mean_dice = np.array(stat['dice']).mean()
    print('Average undersegmentation error: %.4f' % (mean_ue))
    print('Average dice: %.4f' % (mean_dice))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup_id", default='superpixel', help="superpixel id")
    ap.add_argument("--subdir", default='ISIC_noise', help="[ISIC_noise, JSRT_noise]")
    ap.add_argument("--class_name", default='lesion', help= "['lesion', 'lung', 'heart', 'clavicle']")
    args = ap.parse_args()

    Params = {
        'super_dir': '/group/gaozht/Dataset/%s/train/%s/' % (args.subdir, args.sup_id),
        'clean_dir': '/group/gaozht/Dataset/%s/train/label/' % args.subdir,
        'class': args.class_name
    }

    stat()
