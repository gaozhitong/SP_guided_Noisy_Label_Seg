import argparse
from lib.configs.config import config, update_config

def parse_args(description=''):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cfg', default='exp/default.yaml', help='experiment configure file name', type=str)  #
    parser.add_argument('--id', default='exp', type=str, help='Experiment ID')
    parser.add_argument('--weight_path', help='manually specify model weights', type=str, default='')
    parser.add_argument('--demo', help='val/test phase if given value, enable inference', type=str, default='')
    parser.add_argument('--parallel', action='store_true', help='Data parallel, true if using multi GPUs')

    parser.add_argument('--save_preds', help='save images, preds, (gt)', type=bool, default=False)
    parser.add_argument('--seed', help='random seed', type=int, default=0)
    parser.add_argument('--deterministic', help='whether employ deterministic backpropagation', type=int, default=1)

    return parser.parse_args()

# default complete
def default_complete(config, id):
    import os.path as osp
    project_dir = osp.abspath('.')

    if config.data_dir == '':
        config.data_dir = osp.join(project_dir, 'data')
    if config.model_dir == '':  # checkpoints
        config.model_dir = osp.join(project_dir, 'checkpoints/' + id)
    if config.log_dir == '':
        config.log_dir = osp.join(project_dir, 'log')
    if config.tb_dir == '':
        config.tb_dir = osp.join(config.log_dir, 'tb_' + id)

    return config

args = parse_args()
 # pretrain model
update_config(args.cfg)
opt = config
opt = default_complete(opt, args.id)