import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())
from corevectors_dmd.config_cifar100_vgg16 import *

# Our stuff

# overwrite for final evaluation
drill_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/drillers/CIFAR100_VGG16')
# drill_path = Path.cwd()/'../../data/drillers/CIFAR100_vgg16'
drill_name = 'DMD'
 
phs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/peepholes_post_tune/CIFAR100_VGG16')
# phs_path = Path.cwd()/'../../data/peepholes_post_tune/CIFAR100_vgg16'
phs_name = 'peepavg'

# overwrite verbose
magnitude = 0.004

verbose = True

feature_sizes_dmd = {
                'features.2': 64, 
                'features.7': 128, 
                'features.14': 256, 
                'features.21': 512,
                'features.28': 512,
                # 'classifier.3': 4096
                }
















