'''
Crop image & label to a fixed ROI,
Currently only for JSRT dataset
'''

import skimage.io
from glob import glob
import numpy as np
import json
import os
import shutil
import skimage.io

import ipdb
import random
random.seed(0)

def makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def dataset_crop():
    '''
    for all image, label, label_noise, crop
    save to corresponding dirs
    '''
    subdirs = [
        'train/image/', 'train/label_png/',
        'val/image/', 'val/label_png/',
    ]
    # add noisy subdirs
    for alpha in Params['dataset_noise_ratio']:
        for beta in Params['sample_noise_ratio']:
            subdirs.append('train/label_noise_{}_{}_png/'.format(alpha, beta))

    # tranverse all: crop then save (special case: image )
    for subdir in subdirs:
        src_dir = root + subdir
        tar_dir = save_dir + subdir
        makedir(tar_dir)

        for cls in ['clavicle']:
            src_dir_cls = src_dir + cls if 'image' not in subdir else src_dir
            tar_dir_cls = tar_dir + cls
            if rois[cls] is None:   # directly copy
                if os.path.exists(tar_dir_cls):
                    shutil.rmtree(tar_dir_cls)
                shutil.copytree(src_dir_cls, tar_dir_cls)
            else:
                makedir(tar_dir_cls)
                images = sorted(glob(src_dir_cls + '/*.png'))
                roi = rois[cls]
                for img in images:
                    arr = skimage.io.imread(img)
                    arr_crop = arr[roi[0]: roi[1], roi[2]: roi[3]]
                    save_path = tar_dir_cls + '/%s' % img.split('/')[-1]
                    skimage.io.imsave(save_path, arr_crop)


def label_organize():
    '''
    for all label image, convert to json
    note: include val dirs
    :return:
    '''
    from data_process.util import NpEncoder
    def save_json_dir(src_dir, tar_dir, classes=['lung']):
        paths = {}
        for cls in classes:
            paths[cls] = sorted(glob(src_dir + '%s/*.png' % cls))
        length = len(paths[cls])
        for ith in range(length):
            label = {}
            for cls in classes:
                label[cls] = skimage.io.imread(paths[cls][ith], as_gray=True)
                label[cls][label[cls] > 0] = 1

            save_path = tar_dir + paths[cls][ith].split('/')[-1][:-4] + '.json'
            with open(save_path, 'w') as f:
                json.dump(label, f, cls=NpEncoder)

    src_dirs = ['label_png']
    for alpha in Params['dataset_noise_ratio']:
        for beta in Params['sample_noise_ratio']:
            src_dirs.append('label_noise_{}_{}_png'.format(alpha, beta))
    classes = ['lung', 'heart', 'clavicle']

    phases = ['train', 'val']
    for phase in phases:
        if phase == 'train':
            for subdir in src_dirs:
                src_dir = save_dir + phase + '/%s/' % subdir
                tar_dir = save_dir + phase + '/%s/' % subdir[:-4]
                makedir(tar_dir)
                save_json_dir(src_dir, tar_dir, classes=classes)

        elif phase == 'val':
            src_dir = save_dir + phase + '/label_png/'
            tar_dir = save_dir + phase + '/label/'
            makedir(tar_dir)
            save_json_dir(src_dir, tar_dir, classes=classes)



if __name__ == '__main__':
    Params = {
        'dataname': 'jsrt',         #

        'dataset_noise_ratio': [0.3, 0.5, 0.7, 0.9, 1.0],
        'sample_noise_ratio':  [0.5, 0.7],
    }

    # build vars
    roots = {
        'jsrt':     '/group/gaozht/Dataset/JSRT_noise/'
    }
    rois = {
        ## only apply cropping for clavicle
        # 'lung': None,
        # 'heart': None,
        'clavicle': [0, 96, 24, 248],      # (xmin, xmax, ymin, ymax), collected from given noisy labels, detail see stat_jsrt_roi.py
    }

    root = roots[Params['dataname']]
    save_dir = root[:-1] + 'c/'         # c stand for crop, save to new directory
    makedir(save_dir)
    # ipdb.set_trace()

    # 1. crop then save to new dir
    dataset_crop()
    # 3. convert all label to .json
    label_organize()