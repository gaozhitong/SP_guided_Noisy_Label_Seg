from batchgenerators.dataloading import SlimDataLoaderBase
from glob import glob
import os
import numpy as np
import copy
import ipdb
import json
from skimage import io
from lib.configs.parse_arg import opt, args
import random

def normalize(image, to_zero_mean=False):
    '''
    2D image with intensity range (0, 255). Normlize to (0,1)
    '''
    image = image.astype(np.float)
    image = image / 255.0

    if to_zero_mean:
        image = image - 0.5
        image = image / 0.5

    return image

def rotate90(image, label, prob=0.2, extra_labels=None, square=True):
    '''
    Rotate image & label in angle that equals N(0,1,2,3) * 90 degree
    '''
    if random.random() < prob:
        if square:
            k = random.choice([1, 2, 3])
        else:
            k = 2   # only 180 degree for non_square size
        image = np.rot90(image, k, axes=(0, 1)).copy()  # along HW plane
        label = np.rot90(label, k, axes=(0, 1)).copy()
        if extra_labels is not None:
            for ith in range(len(extra_labels)):
                extra_labels[ith] = np.rot90(extra_labels[ith], k, axes=(0, 1)).copy()

    if extra_labels is None:
        return image, label
    else:
        return image, label, extra_labels



class BGDataset(SlimDataLoaderBase):
    '''
    Note it return shape (B, C, H, W...). Don't forget channel dimension.
    '''
    def __init__(self, data_dir, batch_size, phase='train', cls='lung', aug_rot90=False,
                 shuffle=False, seed_for_shuffle=None, infinite=False, return_incomplete=False,
                 num_threads_in_multithreaded=1):
        """
        :param data: datadir or data dictionary
        :param batch_size:
        :param cls: {'lung', 'heart', 'clavicle'}
        :param shuffle: return shuffle order or not
        :param val_mode: at inference phase
        Each iteration: return {'data': ,'seg': }, with shape (B, W, H)
        """
        super(BGDataset, self).__init__(data_dir, batch_size, num_threads_in_multithreaded)
        self.batch_size = batch_size
        self.phase = phase
        self.shuffle = shuffle
        self.infinite = infinite
        self.data_dir = data_dir
        self.cls = cls
        self.aug_rot90 = aug_rot90


        # [Optional] EM style: saved label dir
        self.em_save_pseudo_dir = ''
        if opt.data.em_save_pseudo_dir != '':  # em pseudo dir
            self.em_save_pseudo_dir = opt.data.em_save_pseudo_dir + '%s/' % args.id

        # load sample ids
        self.samples = []
        split_file = data_dir + 'split/%s.txt' % phase
        with open(split_file, 'r') as f:
            for line in f:
                if line.strip() != '':
                    self.samples.append(line.strip())
        ## sample selection for image-level upper bound
        if self.phase == 'train' and opt.data.use_noisy_label and args.demo == '' and opt.data.upper_type == 'image-level':
            postfix = '_iupper/'
            ref_dir = self.data_dir + '%s/label_noise_%.1f_%.1f%s/' % \
                        (self.phase, opt.data.dataset_noise_ratio, opt.data.sample_noise_ratio, postfix)
            samples = sorted(os.listdir(ref_dir))
            self.samples = [sample[:-5] for sample in samples if '.json' in sample and sample[0]!='.']
        ## load extra clean label
        self.load_clean_label = False
        if self.phase == 'train' and opt.data.load_clean_label:    # and args.demo == '' and opt.data.use_noisy_label
            self.load_clean_label = True
        ## load extra superpixel label
        self.load_superpixel_label = False
        if self.phase == 'train' and opt.data.load_superpixel_label:  # and args.demo == ''
            self.load_superpixel_label = True

        # inner variables
        self.indices = list(range(len(self.samples)))
        seed_for_shuffle = args.seed
        self.rs = np.random.RandomState(seed_for_shuffle)
        self.current_position = None
        self.was_initialized = False
        self.return_incomplete = return_incomplete
        self.last_reached = False
        self.number_of_threads_in_multithreaded = 1

    def __len__(self):
        return len(self.samples)//self.batch_size

    def reset(self):
        assert self.indices is not None
        self.current_position = self.thread_id * self.batch_size
        self.was_initialized = True
        self.rs.seed(self.rs.randint(0, 999999999))
        if self.shuffle:
            self.rs.shuffle(self.indices)
        self.last_reached = False

    def get_indices(self):
        if self.last_reached:
            self.reset()
            raise StopIteration

        if not self.was_initialized:
            self.reset()

        if self.infinite:
            return np.random.choice(self.indices, self.batch_size, replace=True, p=None)

        indices = []

        for b in range(self.batch_size):
            if self.current_position < len(self.indices):
                indices.append(self.indices[self.current_position])
                self.current_position += 1
            else:
                self.last_reached = True
                break

        if len(indices) > 0 and (not self.last_reached or self.return_incomplete):
            self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
            return indices
        else:
            self.reset()
            raise StopIteration

    def generate_train_batch(self):
        # similar to __getiterm__(index), but not index as params
        indices = self.get_indices()
        data = {'image': [], 'label': []}
        if self.load_clean_label:
            data['clean_label'] = []
        if self.load_superpixel_label:
            data['superpixel'] = []

        for ith, index in enumerate(indices):
            sample_id = self.samples[index]
            # 1. data path, load data
            paths = {
                'image': self.data_dir + '%s/image/%s.png' % (self.phase, sample_id),
                'label': self.data_dir + '%s/label/%s.json' % (self.phase, sample_id)
            }
            if opt.data.crop_style:     # use cropped image
                paths['image'] = self.data_dir + '%s/image/%s/%s.png' % (self.phase, opt.data.dataname, sample_id)\

            if self.phase == 'train' and self.load_superpixel_label:
                sp_name = opt.data.sp_name      # default 'superpixel'
                paths['superpixel'] = self.data_dir + '%s/%s/%s.json' % (self.phase, sp_name, sample_id)

            if self.load_clean_label:
                paths['clean_label'] = paths['label']

            # noisy label setting
            if self.phase == 'train' and opt.data.use_noisy_label:  #  and args.demo==''
                # load clean label, key 'clean_label'
                # load noisy label, key 'label'
                paths['label'] = self.data_dir + '%s/label_noise_%.1f_%.1f/%s.json' % \
                                 (self.phase, opt.data.dataset_noise_ratio, opt.data.sample_noise_ratio, sample_id)
                # load noisy label from given label dir (only for EM try)
                if opt.data.load_label_dir != '':
                    paths['label'] = opt.data.load_label_dir + '%s.json' % (sample_id)

                # upper bound of noisy label setting
                if opt.data.upper_type in ['pixel-level', 'image-level']:
                    postfix = '_pupper/' if opt.data.upper_type == 'pixel-level' else '_iupper/'
                    paths['label'] = self.data_dir + '%s/label_noise_%.1f_%.1f%s/%s.json' % \
                        (self.phase, opt.data.dataset_noise_ratio, opt.data.sample_noise_ratio, postfix, sample_id)
                # EM style: saved pseudo label
                if (self.em_save_pseudo_dir != ''):
                    em_pseudo_path = self.em_save_pseudo_dir + '%s.json' % sample_id
                    if os.path.exists(em_pseudo_path):
                        paths['label'] = em_pseudo_path

            # image related param
            image_as_gray = True
            to_zero_mean = False
            if opt.model.input_channel == 3:
                image_as_gray = False
                to_zero_mean = True

            image = io.imread(paths['image'], as_gray=image_as_gray)
            image = normalize(image, to_zero_mean)

            with open(paths['label'], 'r') as f:
                label = json.load(f)[self.cls]
                label = np.array(label)
            if self.load_clean_label:
                with open(paths['clean_label'], 'r') as f:
                    clean_label = json.load(f)[self.cls]
                    clean_label = np.array(clean_label)

            if self.load_superpixel_label:
                with open(paths['superpixel'], 'r') as f:
                    superpixel = json.load(f)
                    superpixel = np.array(superpixel)

            # augmentation: rotate 90
            if self.phase == 'train' and self.aug_rot90:
                square = True if not opt.data.crop_style else False
                if self.load_clean_label or self.load_superpixel_label:
                    extra_labels = []
                    if self.load_clean_label:
                        extra_labels.append(clean_label)
                    if self.load_superpixel_label:
                        extra_labels.append(superpixel)
                    image, label, extra_labels = rotate90(image, label, extra_labels=extra_labels, square=square)
                    if self.load_clean_label:
                        clean_label = extra_labels.pop(0)
                    if self.load_superpixel_label:
                        superpixel = extra_labels.pop(0)
                else:
                    image, label = rotate90(image, label, square=square)

            # 3. expand channel dimension
            if opt.model.input_channel == 3:    # to shape (C, H, W)
                data['image'].append(np.transpose(image, (2, 0, 1)))
            else:
                data['image'].append(np.expand_dims(image, 0))
            data['label'].append(np.expand_dims(label, 0))
            if self.load_clean_label:
                data['clean_label'].append(np.expand_dims(clean_label, 0))
            if self.load_superpixel_label:
                data['superpixel'].append(np.expand_dims(superpixel, 0))

        for key, value in data.items():
            data[key] = np.array(value)
        data['data'] = data.pop('image')
        data['seg'] = data.pop('label')
        return data


if __name__ == '__main__':
    # Params
    data_dir = '/group/gaozht/Dataset/JSRT_noise/'
    batch_size = 2
    shuffle = False #True
    phase = 'train' #'train'
    cls = 'lung'
    aug_rot90 = True

    # dataset = BGDataset(data_dir, batch_size, phase=phase, cls=cls, shuffle=shuffle, aug_rot90=aug_rot90)
    dataset = BGDataset(opt.data.data_dir, batch_size, phase=phase, cls=opt.data.dataname, shuffle=shuffle,
                        aug_rot90=aug_rot90)
    print(len(dataset))

    # save some images
    import matplotlib as mpl; mpl.use('Agg')
    import matplotlib.pyplot as plt
    save_dir = '/group/gaozht/nlseg_exp/output/verify_dataset_noise/'

    for ith, batch in enumerate(dataset):
        # print(batch.keys())
        # print(batch['data'].shape)
        # print(batch['seg'].shape)
        # ipdb.set_trace()
        image, label = batch['data'], batch['seg']
        for jth in range(image.shape[0]):
            img, lbl = image[jth, 0], label[jth, 0]
            # img, lbl = image[jth], label[jth, 0]  # for 3-channel image
            # img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            plt.savefig(save_dir + 'train_image%01d%01d.png' % (ith, jth))
            plt.cla()
            plt.imshow(lbl)
            plt.savefig(save_dir + 'train_label%01d%01d.png' % (ith, jth))
            plt.cla()
        if ith > 3:
            break


