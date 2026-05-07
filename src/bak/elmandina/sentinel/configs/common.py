import sys
import os

import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

#sys.path.insert(0, (Path.home()/'repos/FIORIRE/src').as_posix())

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D


# XPs stuff


input_key = 'data'
emb_size = 'large'
num_sensors = 10 #8layers so 8 features
seq_len = 10
kernel = [3, 2]
lay3 = False
seed = 42
#bs = 2**11
bs = 64
verbose = True
lr = 0.003
lr_patience = 5
epochs = 50
es_patience = 10


#----------------------
#  Directories defs
#----------------------

model_path = Path('/srv/newpenny/XAI/generated_data/FIORIRE/train_cps')
model_name = 'checkpoints.499.pt'

ds_path = '/srv/newpenny/dataset/TASI/sentinel/sentinel_4s_clean_std'
parsed_path = Path.home()/'repos/XAI/data/datasets'


#tune_dir = Path('/srv/newpenny/XAI/generated_data/FIORIRE/train_cps')
tune_dir = Path.home()/'repos/XAI/data/train_cv_cps'
tune_name = 'cv_checkpoints'

output_dir = Path.home()/'repos/XAI/data/cv_model'

model_sd_key = 'model_state_dict'


#-------Loaders----------------------
loaders = ['train', 'val', 'test', 'val-c-all']
#-------------------------------------