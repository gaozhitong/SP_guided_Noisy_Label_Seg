import yaml
from easydict import EasyDict as edict

config = edict()

# 1. data_dir
config.snapshot = 10

config.data_dir = ''
config.model_dir = ''
config.log_dir = ''
config.tb_dir = ''

# 2. data related
config.data = edict()
config.data.data_dir = ''
config.data.dataname = ''

config.data.aug_rot90 = False   # rotate 90 degree, only for 2D image
config.data.num_workers = 4
config.data.write_log_batch = True      # logging per batch

## noisy label related
config.data.use_noisy_label = False
config.data.dataset_noise_ratio = 0.0
config.data.sample_noise_ratio = 0.0
config.data.upper_type = ''
config.data.load_clean_label = False

config.data.crop_style = False  # use cropped image and labels

## superpixel related
config.data.load_superpixel_label = False
config.data.sp_name = 'superpixel'

## EM style
config.data.em_save_pseudo_dir = ''
config.data.load_label_dir = '' # load label from other dir, only used for EM try

# 3. model related
config.model = edict()
config.model.network = 'VNet'
## nnU-net parameter
config.model.pool_op_kernel_sizes = ''
config.model.conv_kernel_sizes = ''
config.model.deep_supervision = False
config.model.use_finetune = False
config.model.finetune_model_path = ''
config.model.input_channel = 1

# 4. training params
config.train = edict()
config.train.n_epochs = 100
config.train.n_batches = 50 # default 250 for nnU-net
config.train.train_batch = 2
config.train.valid_batch = 1
config.train.test_batch = 1

config.train.iterate_epoch = 0  # EM style: update label every some epoch (0 means not employed)
config.train.iterate_elist = None   # iterate at these epochs

config.train.loss_name = ''
config.train.fg_weight = 1
config.train.auto_weight = False
config.train.with_dice_loss = 0

config.train.lr = 1e-2
config.train.lr_decay = 'constant'
config.train.lr_warmup = 1e-2
config.train.min_lr = 0
config.train.momentum = 0.9
config.train.weight_decay = 1e-4

config.train.milestone = ''
config.train.gamma = 0.1
config.train.plateau_patience = 3
config.train.plateau_gamma = 0.1

## dataloader mutli-threads
config.train.num_threads = 4
config.train.num_cached_per_thread = 3

# 5. framework
config.framework = edict()
config.stage_warmup = False

# Configs for noise-ware training
config.framework.warmup_epoch = 0
config.framework.remember_rate_fg = 1.0
config.framework.remember_rate_bg = 1.0
config.framework.update_remember_rate_fg = None
config.framework.update_remember_rate_bg = None

config.framework.co_teaching = False
config.framework.JoCoR = False
config.framework.tri_net = False
config.framework.pixel_select = True
config.framework.co_lambda = 0.65

# Configs for label updating.
config.framework.E_step_epoch = -1           # enable label update if E_step_epoch > 0, at epoch __. (<=0 by default)
config.framework.Auto_E_step_epoch = False
config.framework.E_update_remember_rate = False
config.framework.E_update_remember_rate_linear = False
config.framework.rem_linear_param = 1.1
config.framework.rem_thres =  1
config.framework.rem_thres_bg = 1
config.framework.conf_thres_fg = None
config.framework.conf_thres_bg = None
config.framework.corr_ratio_fg = -1
config.framework.corr_ratio_bg = -1
config.framework.corr_ratio_decay = 1.0
config.framework.not_improved_epochs = 5

# superpixels
config.framework.sp_smoothing_weight = None
config.framework.superpixel_select = False
config.framework.sp_label_update_style = None # label update style {'mean'}
# config.framework.n_segments = 400
# config.framework.compactness = 0.1
# config.framework.sup_T = 5

# update method
def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                config[k] = v

