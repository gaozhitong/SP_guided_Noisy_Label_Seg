'''
Dataset process pipeline
1. train/val split
2. split image/label to train/val
3. generate corresponding label.json
4. Noisy generation
    noisy ratios generation
    noisy log, label .json generation
'''

import skimage.io
import cv2
from glob import glob
import numpy as np
import json
import os
import shutil

import ipdb
import random
random.seed(0)

def makedir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def split_dataset():
    if Params['dataname'] in ['jsrt']:
        # copy files: split, train image/label, val image/label
        src_dirs = {
            'split': root + 'split/',
            'train_image': root + 'train/image/',
            'train_label': root + 'train/label/',
            'train_label_png': root + 'train/label_png/',
            'val': root + 'val/'
        }
        tar_dirs = {
            'split': save_dir + 'split/',
            'train_image': save_dir + 'train/image/',
            'train_label': save_dir + 'train/label/',
            'train_label_png': save_dir + 'train/label_png/',
            'val': save_dir + 'val/'
        }
        for key in src_dirs:
            shutil.copytree(src_dirs[key], tar_dirs[key])

    elif Params['dataname'] in ['shenzhen']:
        # obtain all images
        image_dir = root + 'img/'
        mask_dir = root + 'mask/'
        samples = [name[:-9] for name in os.listdir(mask_dir) if (('.png' in name) and (name[0]!='.'))]
        samples = sorted(samples)
        print('Total %d images in this dataset. (Refer to num=566)' % len(samples))

        # split them
        val_ratio = 0.3
        val_num = round(len(samples) * val_ratio)
        data_names = {}
        data_names['val'] = random.sample(samples, val_num)
        data_names['train'] = [name for name in samples if name not in data_names['val']]
        print('Split done: train num %d, val num %d' % (len(data_names['train']), len(data_names['val'])))

        # save split file
        split_dir = save_dir + 'split/'
        makedir(split_dir)
        for phase in ['train', 'val']:
            with open(split_dir+'%s.txt' % phase, 'w') as f:
                for _id in data_names[phase]:
                    f.write('%s\n' % _id)

        # move images and masks, resized to (256, 256)
        out_shape = (256, 256)
        for phase in ['train', 'val']:
            # make dirs
            makedir(save_dir + phase + '/')
            image_dir, label_dir = save_dir + phase + '/image/', save_dir + phase + '/label_png/'
            makedir(image_dir); makedir(label_dir)

            for name in data_names[phase]:
                # image: copy
                src = root + 'img/' + name + '.png'
                dst = image_dir + name + '.png'
                arr = skimage.io.imread(src, as_gray=True)
                arr = cv2.resize(arr, out_shape, interpolation=cv2.INTER_LINEAR)     # bilinear interpolation
                arr = arr.astype(np.uint8)
                skimage.io.imsave(dst, arr, check_contrast=False)

                # label: copy & rename
                src = root + 'mask/' + name + '_mask.png'
                dst = label_dir + name + '.png'
                arr = skimage.io.imread(src, as_gray=True)
                arr[arr > 0] = 1
                arr = cv2.resize(arr, out_shape, interpolation=cv2.INTER_NEAREST)     # nearest interpolation
                arr[arr > 0] = 255
                skimage.io.imsave(dst, arr, check_contrast=False)

    elif Params['dataname'] in ['isic']:
        # obtain all images (train/test sets have been given)
        data_names = {}
        _label_dirs = {
            'train': root + 'ISIC-2017_Training_Part1_GroundTruth/',
            'val':   root + 'ISIC-2017_Test_v2_Part1_GroundTruth/',
        }
        for phase in ['train', 'val']:
            _dir = _label_dirs[phase]
            samples = os.listdir(_dir)
            samples = [sample[:12] for sample in samples if '.png' in sample and sample[0] != '.']
            data_names[phase] = sorted(samples)
        print('Split: train num %d, val num %d' % (len(data_names['train']), len(data_names['val'])))

        # save split file
        split_dir = save_dir + 'split/'
        makedir(split_dir)
        for phase in ['train', 'val']:
            with open(split_dir + '%s.txt' % phase, 'w') as f:
                for _id in data_names[phase]:
                    f.write('%s\n' % _id)

        # move images and masks, resized to (128, 128)
        out_shape = (128, 128)
        _img_dirs = {
            'train': root + 'ISIC-2017_Training_Data/',
            'val':   root + 'ISIC-2017_Test_v2_Data/',
        }
        for phase in ['train', 'val']:
            # make dirs
            makedir(save_dir + phase + '/')
            image_dir, label_dir = save_dir + phase + '/image/', save_dir + phase + '/label_png/'
            makedir(image_dir); makedir(label_dir)

            for name in data_names[phase]:
                # image: copy & rename
                src = _img_dirs[phase] + name + '.jpg'
                dst = image_dir + name + '.png'
                arr = skimage.io.imread(src)
                arr = cv2.resize(arr, out_shape, interpolation=cv2.INTER_LINEAR)     # bilinear interpolation
                arr = arr.astype(np.uint8)
                skimage.io.imsave(dst, arr, check_contrast=False)

                # label: copy & rename
                src = _label_dirs[phase] + name + '_segmentation.png'
                dst = label_dir + name + '.png'
                arr = skimage.io.imread(src)
                arr[arr > 0] = 1
                arr = cv2.resize(arr, out_shape, interpolation=cv2.INTER_NEAREST)     # nearest interpolation
                arr[arr > 0] = 255
                skimage.io.imsave(dst, arr, check_contrast=False)


def noisy_label_generation():
    from data_process.noise_generation import add_noise

    if Params['dataname'] in ['shenzhen', 'isic']:
        # load all sample ids
        load_dir = save_dir + 'train/image/'
        sample_ids = [_id[:-4] for _id in os.listdir(load_dir) if (_id[0] != '.') and ('.png' in _id)]
        sample_ids = sorted(sample_ids)

        for alpha in Params['dataset_noise_ratio']:
            for beta in Params['sample_noise_ratio']:
                # directory init
                target_dir = os.path.join(save_dir, 'train/label_noise_{}_{}_png/'.format(alpha, beta))
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                makedir(target_dir)

                log_path = save_dir + 'train/noise_{}_{}_log.txt'.format(alpha, beta)
                log = open(log_path, 'w')

                # fix random seed, for consistent shuffle and noise type
                random.seed(0)
                ids = sample_ids.copy()
                random.shuffle(ids)
                noisy_sample_num = int(len(ids) * alpha)
                noisy_ids = ids[:noisy_sample_num]

                # add noise
                for _id in sample_ids:
                    clean_label_path = save_dir + 'train/label_png/%s.png' % _id
                    noisy_label_path = target_dir + '%s.png' % _id
                    clean_label = skimage.io.imread(clean_label_path, as_gray=True)
                    clean_label[clean_label > 0] = 1

                    if _id in noisy_ids:
                        noisy_label, noise_type = add_noise(clean_label, noise_ratio=beta)
                    else:
                        noisy_label, noise_type = clean_label, 'clean'
                    noisy_label[noisy_label > 0] = 255  # for visualize
                    skimage.io.imsave(noisy_label_path, noisy_label, check_contrast=False)
                    log.write('%s\t%s\n' % (_id, noise_type))
                log.close()

    elif Params['dataname'] == 'jsrt':
        # 3 classes
        # load all sample ids
        load_dir = save_dir + 'train/image/'
        sample_ids = [_id[:-4] for _id in os.listdir(load_dir) if (_id[0] != '.') and ('.png' in _id)]
        sample_ids = sorted(sample_ids)

        rotate_angles = {'lung': 15, 'heart': 15, 'clavicle': 2}

        for alpha in Params['dataset_noise_ratio']:
            for beta in Params['sample_noise_ratio']:
                # directory init
                target_dir = os.path.join(save_dir, 'train/label_noise_{}_{}_png/'.format(alpha, beta))
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                makedir(target_dir)
                log_path = save_dir + 'train/noise_{}_{}_log.txt'.format(alpha, beta)
                log = open(log_path, 'w')

                for cls in ['lung', 'heart', 'clavicle']:
                    makedir(target_dir + cls)

                    # fix random seed, for consistent shuffle and noise type
                    random.seed(0)
                    ids = sample_ids.copy()
                    random.shuffle(ids)
                    noisy_sample_num = int(len(ids) * alpha)
                    noisy_ids = ids[:noisy_sample_num]

                    # add noise
                    for _id in sample_ids:
                        clean_label_path = save_dir + 'train/label_png/%s/%s.png' % (cls, _id)
                        noisy_label_path = target_dir + '%s/%s.png' % (cls, _id)
                        clean_label = skimage.io.imread(clean_label_path, as_gray=True)
                        clean_label[clean_label > 0] = 1

                        if _id in noisy_ids:
                            noisy_label, noise_type = add_noise(clean_label, noise_ratio=beta, max_rot_angle=rotate_angles[cls])
                        else:
                            noisy_label, noise_type = clean_label, 'clean'
                        noisy_label[noisy_label > 0] = 255  # for visualize
                        skimage.io.imsave(noisy_label_path, noisy_label, check_contrast=False)

                        if cls == 'lung':
                            log.write('%s\t%s\n' % (_id, noise_type))
                log.close()



def noisy_label_organize():
    from data_process.util import NpEncoder
    def save_json_dir(src_dir, tar_dir, classes=['lung']):
        if classes == ['lung'] or classes == ['lesion']:
            cls = classes[0]
            paths = {
                cls: sorted(glob(src_dir + '*.png'))}
            for ith in range(len(paths[cls])):
                mask = skimage.io.imread(paths[cls][ith], as_gray=True)
                mask[mask > 0] = 1
                label = {cls: mask}

                save_path = tar_dir + paths[cls][ith].split('/')[-1][:-4] + '.json'
                with open(save_path, 'w') as f:
                    json.dump(label, f, cls=NpEncoder)

        elif len(classes) > 1:
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
    classes = []
    if Params['dataname'] in ['shenzhen']:
        classes = ['lung']
    elif Params['dataname'] in ['isic']:
        classes = ['lesion']
    elif Params['dataname'] == 'jsrt':
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
        'dataname': 'jsrt',         #{'jsrt', 'isic', }

        'dataset_noise_ratio': [0.3, 0.5, 0.7, 0.9, 1.0],
        'sample_noise_ratio':  [0.5, 0.7],
    }

    # build vars
    roots = {
        'shenzhen': '/group/gaozht/Dataset/shenzhen/',
        'isic':     '/group/gaozht/Dataset/ISIC/',
        'jsrt':     '/group/gaozht/Dataset/JSRT_noise/'
    }

    root = roots[Params['dataname']]
    save_dir = root[:-1] + '_noise/'
    makedir(save_dir)
    # ipdb.set_trace()

    # 1. train/val split, organize dirs
    split_dataset()
    # ipdb.set_trace()
    # 2. noise generation
    noisy_label_generation()
    # 3. convert all label to .json
    noisy_label_organize()