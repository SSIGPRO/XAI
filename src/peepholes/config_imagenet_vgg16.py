import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())
from tuning_methods.config_imagenet_vgg16 import *

# Our stuff
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

# overwrite for final evaluation
#phs_path = Path('/srv/newpenny/XAI/generated_data/peepholes_post_tune/CIFAR100_vgg16')
phs_path = Path.cwd()/'../../data/CLIP/peepholes/ImageNet_VGG16'
phs_name = 'peepholes'

# overwrite verbose
verbose = True

n_classifier = {
        # 'features.7': 1360,
        # 'features.10': 2239,
        # 'features.12': 2159,
        # 'features.14': 524,
        # 'features.17': 773,
        # 'features.19': 1326,
        # 'features.21': 851,
        # 'model.features.24': 1035,
        # 'model.features.26': 838,
        # 'model.features.28': 1481,
        # 'model.classifier.0': 1717,
        'model.classifier.3': 1627,
        # 'model.classifier.6': 1706,
        }

peep_size = {
        # 'features.7': 500,
        # 'features.10': 479,
        # 'features.12': 442,
        # 'features.14': 496,
        # 'features.17': 356,
        # 'features.19': 499,
        # 'features.21': 489,
        # 'model.features.24': 50,
        # 'model.features.26': 50,
        # 'model.features.28': 50,
        # 'model.classifier.0': 50,
        'model.classifier.3': 53,
        # 'model.classifier.6': 51,
        }

drillers = {}
for _layer in target_layers:
    parser_cv = partial(
            trim_corevectors,
            module = _layer,
            cv_dim = peep_size[_layer],
            )

    drillers[_layer] = tGMM(
            path = drill_path,
            name = drill_name+'.'+_layer,
            nl_classifier = n_classifier[_layer],
            nl_model = 1000,
            n_features = peep_size[_layer], 
            parser = parser_cv,
            device = device
            )
                                                                     
#     drillers[_layer].load()
