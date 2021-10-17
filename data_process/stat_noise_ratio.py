import os
import numpy as np
import argparse
from glob import glob
from sklearn import metrics
import skimage.io
import ipdb

'''
Statistics: noise ratios in each noisy dataset. (We assume noise ratios are given following previous works)
Given noisy dataset & original dataset, calculate clean ratios. (noise ratio = 1 - clean ratio)
'''


def noise_metric(noise_set):
    # load noisy label & gt label
    if subdir == 'ISIC_noise2':
        paths['noisy_label'] = sorted(glob(Params['root'] + 'label_noise_%.1f_%.1f_png/*.png' % (noise_set[0], noise_set[1])))
        all_noisy_images = np.zeros((N, 128, 128), dtype=np.uint8)
    elif subdir in ['JSRT_noise2', 'JSRT_noise3']:
        paths['noisy_label'] = sorted(glob(Params['root'] + 'label_noise_%.1f_%.1f_png/%s/*.png' % (noise_set[0], noise_set[1], cls)))
        all_noisy_images = np.zeros((N, 256, 256), dtype=np.uint8)
    # stack them
    for ith in range(N):
        path_noisy = paths['noisy_label'][ith]
        all_noisy_images[ith] = skimage.io.imread(path_noisy)

    # # calculate precision
    # matrix = metrics.confusion_matrix(all_noisy_images.flatten(), all_clean_images.flatten(), labels=[0, 255],
    #                                   normalize='true')
    #
    # return {
    #     'bg': matrix[0, 0],
    #     'fg': matrix[1, 1],
    # }

    fg_correct = np.logical_and(all_noisy_images > 0, all_clean_images > 0).sum()
    fg_precision = 1.0 * fg_correct / (all_noisy_images > 0).sum()
    bg_correct = np.logical_and(all_noisy_images == 0, all_clean_images == 0).sum()
    bg_precision = 1.0 * bg_correct / (all_noisy_images == 0).sum()
    return {
        'bg': bg_precision,
        'fg': fg_precision,
        'fg/bg': (all_noisy_images > 0).sum() / (all_noisy_images == 0).sum()
    }


if __name__ == '__main__':
    subdir = 'JSRT_noise'      # ISIC_noise
    cls = 'clavicle'
    Params = {
        'root':            '/group/gaozht/Dataset/%s/train/' % subdir,
        'clean_label_dir': '/group/gaozht/Dataset/%s/train/label_png/' % subdir,
        'noise_setting': [[0.3, 0.5], [0.3, 0.7], [0.5, 0.5], [0.5, 0.7],
                          [0.7, 0.5], [0.7, 0.7], [0.9, 0.5], [0.9, 0.7], [1.0, 0.5], [1.0, 0.7]]

    }

    if subdir == 'ISIC_noise2':
        paths = {
            'gt_label': sorted(glob(Params['clean_label_dir'] + '*.png')),
            'noisy_label': '',
        }
        N = (len(paths['gt_label']))
        all_clean_images = np.zeros((N, 128, 128), dtype=np.uint8)
    elif subdir in ['JSRT_noise2', 'JSRT_noise3']:
        paths = {
            'gt_label': sorted(glob(Params['clean_label_dir'] + '%s/*.png' % cls)),
        }
        N = (len(paths['gt_label']))
        all_clean_images = np.zeros((N, 256, 256), dtype=np.uint8)

    for ith in range(N):
        path_clean = paths['gt_label'][ith]
        all_clean_images[ith] = skimage.io.imread(path_clean)

    for noise_set in Params['noise_setting']:
        noise_ratios = noise_metric(noise_set)
        print('noise ratios', noise_set, noise_ratios)
