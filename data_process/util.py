
from skimage import io
from glob import glob
import numpy as np
import json
import os
import ipdb


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def build_split_files():
    root = '/Users/seolen/Seolen-Project/_group/gaozht/Dataset/JSRT_noise/'
    phases = ['train', 'val']
    for phase in phases:
        src_dir = root + phase + '/image/'
        tar_path = root + 'split/%s.txt' % phase
        sample_ids = sorted(os.listdir(src_dir))
        sample_ids = [sample_id[:-4] for sample_id in sample_ids if sample_id[0]!='.']
        with open(tar_path, 'w') as f:
            for sample_id in sample_ids:
                f.write(sample_id + '\n')
