import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())
from tuning_methods.config_cifar100_ViT import *

# Our stuff
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

# overwrite for final evaluation
phs_path = Path('/srv/newpenny/XAI/generated_data/TPAMI/peepholes_post_tune/CIFAR100_ViT')
# phs_path = Path.cwd()/'../../data/peepholes_post_tune/CIFAR100_ViT'
phs_name = 'peepholes'

# overwrite verbose
verbose = True

n_classifier = {
        # 'encoder.layers.encoder_layer_0.mlp.0': 1694,
        # 'encoder.layers.encoder_layer_0.mlp.3': 1417,
        # 'encoder.layers.encoder_layer_1.mlp.0': 2436,
        # 'encoder.layers.encoder_layer_1.mlp.3': 3252,
        # 'encoder.layers.encoder_layer_2.mlp.0': 1527,
        # 'encoder.layers.encoder_layer_2.mlp.3': 1423,
        # 'encoder.layers.encoder_layer_3.mlp.0': 1881,
        # 'encoder.layers.encoder_layer_3.mlp.3': 2921,
        # 'encoder.layers.encoder_layer_4.mlp.0': 2295,
        # 'encoder.layers.encoder_layer_4.mlp.3': 1606,
        # 'encoder.layers.encoder_layer_5.mlp.0': 1761,
        # 'encoder.layers.encoder_layer_5.mlp.3': 1321,
        # 'encoder.layers.encoder_layer_6.mlp.0': 1945,
        # 'encoder.layers.encoder_layer_6.mlp.3': 2381,
        # 'encoder.layers.encoder_layer_7.mlp.0': 1234,
        # 'encoder.layers.encoder_layer_7.mlp.3': 1766,
        # 'encoder.layers.encoder_layer_8.mlp.0': 1025,
        # 'encoder.layers.encoder_layer_8.mlp.3': 1486,
        # 'encoder.layers.encoder_layer_9.mlp.0': 918,
        # 'encoder.layers.encoder_layer_9.mlp.3': 1263,
        # 'encoder.layers.encoder_layer_10.mlp.0': 1060,
        # 'encoder.layers.encoder_layer_10.mlp.3': 2473,
        'encoder.layers.encoder_layer_11.mlp.0': 3714,
        # 'encoder.layers.encoder_layer_11.mlp.3': 2373,
        # 'heads.head': 4519,
        }

peep_size = {
        # 'encoder.layers.encoder_layer_0.mlp.0': 448,      
        # 'encoder.layers.encoder_layer_0.mlp.3': 54,
        # 'encoder.layers.encoder_layer_1.mlp.0': 52,
        # 'encoder.layers.encoder_layer_1.mlp.3': 83,
        # 'encoder.layers.encoder_layer_2.mlp.0': 390,
        # 'encoder.layers.encoder_layer_2.mlp.3': 469,
        # 'encoder.layers.encoder_layer_3.mlp.0': 100,
        # 'encoder.layers.encoder_layer_3.mlp.3': 333,
        # 'encoder.layers.encoder_layer_4.mlp.0': 156,
        # 'encoder.layers.encoder_layer_4.mlp.3': 405,
        # 'encoder.layers.encoder_layer_5.mlp.0': 174,
        # 'encoder.layers.encoder_layer_5.mlp.3': 398,
        # 'encoder.layers.encoder_layer_6.mlp.0': 254,
        # 'encoder.layers.encoder_layer_6.mlp.3': 410,
        # 'encoder.layers.encoder_layer_7.mlp.0': 253,
        # 'encoder.layers.encoder_layer_7.mlp.3': 479,
        # 'encoder.layers.encoder_layer_8.mlp.0': 376,
        # 'encoder.layers.encoder_layer_8.mlp.3': 346,
        # 'encoder.layers.encoder_layer_9.mlp.0': 500,
        # 'encoder.layers.encoder_layer_9.mlp.3': 441,
        # 'encoder.layers.encoder_layer_10.mlp.0': 258,
        # 'encoder.layers.encoder_layer_10.mlp.3': 403,
        'encoder.layers.encoder_layer_11.mlp.0': 378,
        # 'encoder.layers.encoder_layer_11.mlp.3': 153,
        # 'heads.head': 52,
        }

drillers = {}
for _layer in target_layers:
    
    parser_cv = partial(
            trim_corevectors,
            module = _layer,
            cv_dim = peep_size[_layer],
            )
    print(parser_cv)

    drillers[_layer] = tGMM(
            path = drill_path,
            name = drill_name+'.'+_layer,
            nl_classifier = n_classifier[_layer],
            nl_model = 100,
            n_features = peep_size[_layer], 
            parser = parser_cv,
            device = device
            )
                                                                     
    drillers[_layer].load()
