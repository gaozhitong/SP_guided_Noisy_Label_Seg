'''
Statistics: ROI calculation of JSRT (class heart & clavicle)
'''

import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import skimage.io
import argparse
import ipdb
import datetime

def calculate_roi(root, cls='clavicle'):
    roi = [1e6, 0, 1e6, 0]      # (xmin, xmax, ymin, ymax)
    paths = sorted(glob(root + '*.json'))
    for path in paths:
        with open(path, 'rb') as f:
            mask = json.load(f)[cls]
        mask = np.array(mask)

        nonzero = np.nonzero(mask)
        xmin, xmax, ymin, ymax = nonzero[0].min(), nonzero[0].max(), nonzero[1].min(), nonzero[1].max()
        roi = [min(roi[0], xmin), max(roi[1], xmax), min(roi[2], ymin), max(roi[3], ymax)]
    return roi


if __name__ == '__main__':
    Params = {
        'clean_label_dir': '/Users/seolen/Seolen-Project/_group/gaozht/Dataset/JSRT_noise/train/label/',
        'noise_label_dir': '/Users/seolen/Seolen-Project/_group/gaozht/Dataset/JSRT_noise/train/label_noise_1.0_0.7/',
    }
    classes = ['clavicle']
    for cls in classes:
        # roi = calculate_roi(Params['clean_label_dir'], cls=cls)
        # print('gt', cls, roi)
        roi = calculate_roi(Params['noise_label_dir'], cls=cls)
        print('noise', cls, roi)

'''
(xmin, xmax, ymin, ymax)
** noise from aXb7
• gt clavicle  [7, 85, 40, 230]     # only for observing, don't use it 
• gt heart     [84, 241, 68, 223]   # only for observing, don't use it 
• noise clavicle  [0, 91, 13, 241]  
• noise heart     [72, 255, 57, 250]
'''