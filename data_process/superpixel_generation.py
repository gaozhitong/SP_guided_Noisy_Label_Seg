'''
Generate superpixel labels for training images
'''


from skimage.segmentation import slic, mark_boundaries
from skimage.io import imread, imsave
import skimage.exposure
from glob import glob
import json
from jsrt_organize import NpEncoder #data_process.
import os
import numpy as np
import argparse



import ipdb

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--super_postfix', default= '', help='superpixel dir post fix', type=str)
    args = parser.parse_args()

    return args

def makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def gen_superpixel_label(paths):
    def save_json(save_path, segments):
        with open(save_path, 'w') as f:
            json.dump(segments, f, cls=NpEncoder)
    def save_img_overlay(save_path, boundary):
        imsave(save_path, boundary, check_contrast=False)

    for path in paths:
        name = path.split('/')[-1]
        img = imread(path)
        if apply_contrast:
            img = skimage.exposure.equalize_adapthist(img)

        if use_prob:
            prob_t5 = np.expand_dims(np.load(os.path.join(Params['root_prob'], name.split('.')[0] + '.npy')), 2).astype(
                'double') * 255
            # We use softmax with temperature 5 as learned features to help generate superpixels.

            segments = slic(np.concatenate(( np.expand_dims(img,2), prob_t5), axis = 2), n_segments=n_segments, compactness=compactness)  # shape (H, W)
        else:
            segments = slic(img, n_segments=n_segments, compactness=compactness)    # shape (H, W)

        # save superpixel label
        save_path = Params['tar_dir'] + name.split('.png')[0] + '.json'
        save_json(save_path, segments)

        # save visualization: superpixel boundary on image
        boundary = mark_boundaries(img, segments, mode='thick')
        save_path = Params['tar_dir2'] + name
        save_img_overlay(save_path, boundary)



if __name__ == '__main__':
    subdir = 'ISIC_noise'     # 'JSRT_noise'
    args = parse_args()

    Params = {
        'root': '/group/gaozht/Dataset/%s/train/image/' % subdir,
        # 'root_prob': '/group/gaozht/nlseg_exp/output/%s_train/heatmap_npy' %args.super_postfix,
        'tar_dir': '/group/gaozht/Dataset/%s/train/superpixel_%s/' % (subdir, args.super_postfix),
        'tar_dir2': '/group/gaozht/Dataset/%s/train/superpixel_vis_%s/' % (subdir,args.super_postfix),
    }
    makedir(Params['tar_dir'])
    makedir(Params['tar_dir2'])

    apply_contrast = False
    use_prob = False

    n_segments, compactness = 100, 10   # default param for ISIC
    # n_segments, compactness = 800, 10 # default param for JSRT lung, heart
    # n_segments, compactness = 1200, 10 # default param for JSRT clavicle

    paths = sorted(glob(Params['root'] + '*.png'))
    print('Total images:', len(paths))
    gen_superpixel_label(paths)
