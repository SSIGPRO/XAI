import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())
from corevectors.config_imagenet_vgg16 import *

# drill_path = Path('/srv/newpenny/XAI/generated_data/drillers/ImageNet_vgg16')
drill_path =Path.cwd()/'../../data/CLIP/drillers/ImageNet_VGG16'
drill_name = 'classifier'
                                                
# phs_path = Path('/srv/newpenny/XAI/generated_data/peepholes/ImageNet_vgg16')
phs_path =Path.cwd()/'../../data/peepholes/ImageNet_VGG16'
phs_name = 'peepholes'
                                                
# tune_storage_path = Path('/srv/newpenny/XAI/generated_data/tuning/ImageNet_vgg16')
tune_storage_path =Path.cwd()/'../../data/tuning/ImageNet_VGG16'

# Overwrite batch_size
bs = 2**11

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
    max_peep_size[_l] = 500 if _l != 'classifier.6' else 100
