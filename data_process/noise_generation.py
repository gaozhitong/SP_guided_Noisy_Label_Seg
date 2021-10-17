import random
import os
import shutil
import skimage.io
import scipy.ndimage
import skimage.transform
import numpy as np
import ipdb
random.seed(0)

def dilate_mask(label, noise_ratio=0.2):
    '''
    Add noise by dilation: according to object foreground ratio
    '''
    # total fg num and noisy num
    max_num = label.shape[0] * label.shape[1]
    total_fg_num = (label > 0).sum()
    noisy_num = int(total_fg_num * noise_ratio)
    threshold_num = total_fg_num + noisy_num

    # iterately dilate until exceeding threshold
    noisy_label = label.copy()
    while (noisy_label.sum() < threshold_num and noisy_label.sum() != max_num):
        last_label = noisy_label.copy()
        noisy_label = scipy.ndimage.binary_dilation(noisy_label, iterations=1)
    noisy_label = noisy_label.astype(np.uint8)
    last_label = last_label.astype(np.uint8)

    # choose noisy label with nearest ratio
    if noisy_label.sum() == max_num:
        noisy_label = last_label
    elif abs(last_label.sum() - threshold_num) < abs(noisy_label.sum() - threshold_num):
        noisy_label = last_label
    assert (noisy_label.sum() > 0)

    return noisy_label


def erode_mask(label, noise_ratio=0.2):
    '''
    Add noise by erosion: according to object foreground ratio
    '''
    # total fg num and noisy num
    total_fg_num = (label > 0).sum()
    noisy_num = int(total_fg_num * noise_ratio)
    threshold_num = total_fg_num - noisy_num

    # iterately dilate until exceeding threshold
    noisy_label = label.copy()
    while(noisy_label.sum() > threshold_num and noisy_label.sum() != 0):
        last_label = noisy_label.copy()
        noisy_label = scipy.ndimage.binary_erosion(noisy_label, iterations=1)
    noisy_label = noisy_label.astype(np.uint8)
    last_label = last_label.astype(np.uint8)

    # choose noisy label with nearest ratio
    if noisy_label.sum() == 0:
        noisy_label = last_label
    elif abs(last_label.sum() - threshold_num) < abs(noisy_label.sum() - threshold_num):
        noisy_label = last_label
    assert (noisy_label.sum() > 0)

    return noisy_label


def affine(label, noise_ratio=0.2, max_step=100, max_angle=30):
    '''
    rotation then translation
    '''

    def translate_label(label, noisy_label, step, tri_func=(1, 0)):
        step_x, step_y = step * tri_func[0], step * tri_func[1]
        W, H = label.shape
        new_label = np.zeros((W + abs(step_x), H + abs(step_y)), dtype=label.dtype)

        # origins in new_label
        origin_before = [step_x, step_y]
        if step_x < 0:
            origin_before[0] = abs(step_x)
        if step_y < 0:
            origin_before[1] = abs(step_y)
        new_label[origin_before[0]: origin_before[0] + W, origin_before[1]: origin_before[1] + H] = noisy_label
        new_label = new_label[:W, :H]

        # calculate noise_rate
        noisy_num = np.logical_and(new_label == 1, label == 0).sum() + np.logical_and(new_label == 0, label == 1).sum()
        noise_rate = 1.0 * noisy_num / label.sum()
        return new_label, noise_rate

    # rotation angle
    angle = random.uniform(-max_angle, max_angle)
    rotated_label = skimage.transform.rotate(label.astype(np.float), angle).astype(np.uint8)

    # translate direction: (cos, sin)
    tri_funcs = [(1, 0), (1, 1), (0, 1), (-1, 1),
                 (-1, 0), (-1, 1), (0, -1), (1, -1)]
    tri_func = random.choice(tri_funcs)

    # translate step: divide method
    left, right = 0, max_step
    max_count = 20; count = 0
    while (left < right):
        count += 1
        if count > max_count:
            break

        middle = (left + right) // 2
        noisy_label, noise_metric = translate_label(label, rotated_label, middle, tri_func)
        if noise_metric > noise_ratio:
            right = middle - 1
        elif noise_metric < noise_ratio:
            left = middle + 1
        else:
            break

    return noisy_label


def add_noise(label, noise_ratio=0.2, max_rot_angle=30):
    assert len(label.shape) == 2
    noise_types = ['dilate', 'erode', 'affine']
    noise_type = random.choice(noise_types)
    if noise_type == 'dilate':
        noisy_label = dilate_mask(label, noise_ratio)
    elif noise_type == 'erode':
        noisy_label = erode_mask(label, noise_ratio)
    elif noise_type == 'affine':
        noisy_label = affine(label, noise_ratio, max_angle=max_rot_angle)

    return noisy_label, noise_type

def noisy_label_generation():
    # load all sample ids
    load_dir = Params['root_dir'] + Params['clean_dir'] + 'lung/'
    sample_ids = [_id for _id in os.listdir(load_dir) if (_id[0]!='.') and ('.png' in _id)]
    sample_ids = sorted(sample_ids)
    sample_ids = [_id[:-4] for _id in sample_ids]


    for alpha in Params['dataset_noise_ratio']:
        for beta in Params['sample_noise_ratio']:
            target_dir = os.path.join(Params['root_dir'],  'label_noise_{}_{}_png/'.format(alpha, beta))

            # directory init
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                for subdir in Params['classes']:
                    os.makedirs(target_dir+subdir+'/')

            # subdir: for each class
            for subdir in Params['classes']:
                class_dir = target_dir+subdir+'/'
                log_path = target_dir + '%s_noise_log.txt' % subdir
                log = open(log_path, 'w')


                # fix random seed, for consistent shuffle and noise type
                random.seed(0)
                ids = sample_ids.copy()
                random.shuffle(ids)
                noisy_sample_num = int(len(ids) * alpha)
                noisy_ids = ids[:noisy_sample_num]

                # add noise
                for _id in sample_ids:
                    clean_label_path = Params['root_dir'] + Params['clean_dir'] + '%s/%s.png' % (subdir, _id)
                    noisy_label_path = class_dir + '%s.png' % _id
                    clean_label = skimage.io.imread(clean_label_path, as_gray=True)
                    clean_label[clean_label>0] = 1

                    if _id in noisy_ids:
                        noisy_label, noise_type = add_noise(clean_label, noise_ratio=beta)
                    else:
                        noisy_label, noise_type = clean_label, 'clean'
                    noisy_label[noisy_label>0] = 255    # for visualize
                    skimage.io.imsave(noisy_label_path, noisy_label, check_contrast=False)
                    log.write('%s\t%s\n' % (_id, noise_type))
                log.close()





if __name__ == '__main__':
    # params
    Params = {
        'classes': ['lung', 'heart', 'clavicle'],
        'root_dir': '/group/gaozht/Dataset/JSRT_noise/train/',
        'clean_dir': 'label_png/',
        'dataset_noise_ratio': [0.9, 1.0], # [0.3, 0.5, 0.7],
        'sample_noise_ratio': [0.5, 0.7],
    }

    noisy_label_generation()

