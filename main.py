import os
from lib.utils import random_init
from lib.nlss import NLSS
from lib.configs.config import config, update_config
from lib.configs.parse_arg import default_complete
from lib.configs.parse_arg import opt, args

if __name__ == '__main__':
    # make default dirs
    random_init(args.seed)

    if args.cfg != 'exp/default.yaml':

        # update config
        update_config(args.cfg)
        opt = config
        opt = default_complete(opt, args.id)

        # mkdir
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)

        # Init the whole model.
        nlss_seg = NLSS()

        # Training
        if args.demo == '':
            if opt.framework.co_teaching:
                nlss_seg.noise_aware_training()
            else:
                nlss_seg.train()
        # Inference
        else:
            nlss_seg.inference()
