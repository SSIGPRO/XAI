from pathlib import Path as Path

# import configs
from configs.imagenet import *
from configs.vgg import *

ds_path = Path('../../data')/'datasets'

#--------------------------------
# Running
#--------------------------------
n_threads = 4
bs_base = 2**10
bs_atk_scale = 2**-4
tune_num_samples = 50
seed = 29
