'''
Calculated the dice of saved label compare to GT.
- format: ***.json
'''

import json
import numpy as np
from glob import glob
import argparse
from tqdm import tqdm


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

def stat_dice():
    stat = {'dice': []}
    paths = sorted(glob(Params['json_dir'] + '*.json'))
    gt_paths = sorted(glob(Params['clean_dir'] + '*.json'))
    for ith in tqdm(range(len(paths))):
        path, gt_path = paths[ith], gt_paths[ith]
        with open(path, 'rb') as f:
            mask = json.load(f)[Params['class']]
        mask = np.array(mask)
        with open(gt_path, 'rb') as f:
            gt_mask = json.load(f)[Params['class']]
        gt_mask = np.array(gt_mask)

        stat['dice'].append(dice_func(mask, gt_mask))

    mean_dice = np.array(stat['dice']).mean()
    print('Average dice of %s: %.4f' % (args['id'], mean_dice))



if __name__ == '__main__':
    subdir = 'ISIC_noise'     # ISIC_noise, JSRT_noise
    Params = {
        'json_dir': '/group/gaozht/nlseg_exp/em_save_pseudo/0104_skin_a7b7_spem02_sm01/',

        'clean_dir': '/group/gaozht/Dataset/%s/train/label/' % subdir,
        'class':     'lesion',     # 'lesion', 'lung', 'heart', 'clavicle'
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default='', help="experiment id")
    ap.add_argument("--subdir", default='', help="subdir")
    args = vars(ap.parse_args())
    if args['id'] != '':
        Params['json_dir'] = '/group/gaozht/nlseg_exp/em_save_pseudo/%s/' % args['id']
    if args['subdir'] != '':
        Params['clean_dir'] = '/group/gaozht/Dataset/%s/train/label/' % args['subdir']
    Params['class'] =  args['id'].split('_')[1]
    if Params['class'] == 'clavi':
        Params['class'] = 'clavicle'
    elif 'skin' in Params['class']:
        Params['class'] = 'lesion'

    stat_dice()
