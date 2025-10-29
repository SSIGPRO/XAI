import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())
from corevectors_dmd.config_cifar100_ViT import *

# Our stuff

# overwrite for final evaluation
drill_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/drillers/CIFAR100_ViT')
# drill_path = Path.cwd()/'../../data/drillers/CIFAR100_ViT'
drill_name = 'DMD'

phs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/peepholes_post_tune/CIFAR100_ViT')
# phs_path = Path.cwd()/'../../data/peepholes_post_tune/CIFAR100_ViT'
phs_name = 'peepavg'

# overwrite verbose
magnitude = 0.004

verbose = True

feature_sizes_dmd = { f'encoder.layers.encoder_layer_{i}.mlp.3': 768 for i in range(12)}
# feature_sizes[pl_layer] = 768

