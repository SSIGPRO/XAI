import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())
from corevectors.config_cifar100_ViT import *

drill_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/drillers/CIFAR100_ViT')
#drill_path = Path.cwd()/'../data/drillers/CIFAR100_ViT'
drill_name = 'classifier'
                                                
phs_path = Path('/srv/newpenny/XAI/generated_data/peepholes/CIFAR100_ViT')
#phs_path = Path.cwd()/'../data/peepholes/CIFAR100_ViT'
phs_name = 'peepholes'

tune_storage_path = Path('/srv/newpenny/XAI/generated_data/tuning/CIFAR100_ViT')
#tune_storage_path = Path.cwd()/'../data/tuning/CIFAR100_ViT'

# Overwrite batch_size
bs = 2**12

# Ray Tune
num_samples = 100

# Overwrite verbose
verbose=False

min_n_classifier = 500
max_n_classifier = 5000

min_peep_size = 50
max_peep_size = {}
# set a maximum value for cv_size,  for the last layer max_size = 100
for _l in target_layers:
    max_peep_size[_l] = 500 if _l != 'heads.head' else 100
