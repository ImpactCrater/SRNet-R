from easydict import EasyDict as edict
import json
from os.path import expanduser

config = edict()
config.TRAIN = edict()
config.VALID = edict()

# home path
config.home_path = expanduser("~")

# checkpoint location
config.checkpoint_path = config.home_path + '/SRNet/checkpoint/'

# samples location
config.samples_path = config.home_path + '/SRNet/samples/'

# log location
config.VALID.logdir = config.home_path + '/SRNet/logs/'

## Adam
config.TRAIN.sample_batch_size = 25
config.TRAIN.batch_size = 16
config.TRAIN.learning_rate = 1e-4
config.TRAIN.beta1 = 0.9

## Generator
config.TRAIN.n_epoch = 1000

## train set location
config.TRAIN.hr_img_path = config.home_path + '/SRNet/HRImage_Training/'

## test set location
config.VALID.hr_img_path = config.home_path + '/SRNet/HRImage_Validation/'
config.VALID.eval_img_path = config.home_path + '/SRNet/LRImage_Evaluation/'
config.VALID.eval_img_name = '1.png'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
