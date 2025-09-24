import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff 

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
args = parser.parse_args()

if sys.argv[1] == 'vgg':
    from config.config_vgg import *
elif sys.argv[1] == 'vit':
    from config.config_vit import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg_cifar10|vgg_cifar100|vit_cifar100>\'')


if __name__ == "__main__":
    
    #--------------------------------
    # Directories definitions
    #--------------------------------

    # model parameters 
    seed = 29
    bs = 2**7 
    n_threads = 1

    cvs_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../../data/{model_name}/drillers'
    drill_name = 'classifier'
    
    verbose = True 
    n_examples = 10

    print('using', n_examples, 'for evaluation')
    dataset_ = BenchmarkDataset(dataset)
    threat_model_ = ThreatModel('L2')

    prepr = get_preprocessing(dataset_, threat_model_, model_name,
                                    preprocessing=ds_transform)

    clean_x_test, clean_y_test = load_clean_dataset(dataset_, n_examples,
                                                        ds_path, prepr)

    accuracy = clean_accuracy(model._model,
                                clean_x_test,
                                clean_y_test,
                                batch_size=bs,
                                device=device)
    print(f'Model: {model_name}, Clean accuracy: {accuracy:.2%}')

    print('AutoAttack on', dataset, 'with epsilon 4/ 255')
    adversary = AutoAttack(model._model,
                            norm=threat_model_.value,
                            eps=4/255,
                            version='standard',
                            device=device,
                            )
    x_adv = adversary.run_standard_evaluation(clean_x_test,
                                                clean_y_test,
                                                bs=bs,
                                                state_path=None)

    adv_accuracy = clean_accuracy(model._model,
                                x_adv,
                                clean_y_test,
                                batch_size=bs,
                                device=device)

    print(f'AutoAttack robust accuracy: {adv_accuracy:.2%}')
    