from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['vgg','mobilenet'], default='vgg')
parser.add_argument('-r', '--reduction', choices=['maxp', 'toeplitz', 'kernel'], default='kernel')
parser.add_argument('-a', '--analysis', choices=['macs', 'dmd'], default='macs')
args = parser.parse_args()

# import configs
from configs.common import *

if args.model == 'vgg':
    from configs.vgg import *
elif args.model == 'mobilenet':
    from configs.mobilenet import *

if args.reduction == 'maxp':
    from configs.maxpooling import *
elif args.reduction == 'toeplitz':
    from configs.toeplitz import *
elif args.reduction == 'kernel':
    from configs.kernel import *

if args.analysis == 'macs':
    from configs.macs import *
elif args.analysis == 'dmd':
    from configs.dmd import *

#--------------------------------
# datasets
#--------------------------------

cifar_path = '/srv/newpenny/dataset/CIFAR100'
cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
svhn_path = '/srv/newpenny/dataset/SVHN' 
places_path = '/srv/newpenny/dataset/Places365'

#--------------------------------
# Runing
#--------------------------------
seed = 2
n_threads = 1
verbose = True 

#--------------------------------
# Defs 
#--------------------------------
loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        'CIFAR100-C-val-c0',
        'CIFAR100-C-test-c0',
        'CIFAR100-C-val-c1',
        'CIFAR100-C-test-c1',
        'CIFAR100-C-val-c2',
        'CIFAR100-C-test-c2',
        'CIFAR100-C-val-c3',
        'CIFAR100-C-test-c3',
        'CIFAR100-C-val-c4',
        'CIFAR100-C-test-c4',
        'SVHN-val',
        'SVHN-test',
        'Places365-val',
        'Places365-test',
        'CW-CIFAR100-val',
        'CW-CIFAR100-test',
        'BIM-CIFAR100-val',
        'BIM-CIFAR100-test',
        'DF-CIFAR100-val',
        'DF-CIFAR100-test',
        'PGD-CIFAR100-val',
        'PGD-CIFAR100-test',
        ]
