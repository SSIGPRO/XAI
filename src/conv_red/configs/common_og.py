from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['vgg','mobilenet'], default='vgg')
parser.add_argument('-r', '--reduction', choices=['avgpooling', 'toeplitz', 'kernel'], default='kernel')
parser.add_argument('-a', '--analysis', choices=['macs', 'dmd'], default='macs')
parser.add_argument('-d', '--data_path', default=Path.cwd()/'../../data')
args = parser.parse_args()

if args.model == 'vgg':
    from configs.vgg import *
elif args.model == 'mobilenet':
    from configs.mobilenet import *

if args.reduction == 'avgpooling':
    from configs.avgpooling import *
elif args.reduction == 'toeplitz':
    from configs.toeplitz import *
elif args.reduction == 'kernel':
    from configs.kernel import *

if args.analysis == 'macs':
    from configs.macs import *
elif args.analysis == 'dmd':
    from configs.dmd import *

#--------------------------------
# Paths and Definitions 
#--------------------------------
cifar_path = '/srv/newpenny/dataset/CIFAR100'
cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
svhn_path = '/srv/newpenny/dataset/SVHN' 
places_path = '/srv/newpenny/dataset/Places365'

ds_path = Path(args.data_path)/args.model/'datasets'

svds_path = Path(args.data_path)/args.model/'svds'/args.reduction

cvs_path = Path(args.data_path)/args.model/'corevectors'/args.reduction
cvs_name = 'cvs' 

drill_path = Path(args.data_path)/args.model/'drillers'/args.reduction/args.analysis
drill_name = 'driller' 

phs_path = Path(args.data_path)/args.model/'peepholes'/args.reduction/args.analysis
phs_name = 'phs' 

#--------------------------------
# Running
#--------------------------------
seed = 2
n_threads = 1
verbose = True 
n_classes = 100
bs_base = 2**12
bs_atk_scale = 2**-4

#--------------------------------
# Defs 
#--------------------------------
loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
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
        #'DF-CIFAR100-val',
        #'DF-CIFAR100-test',
        #'PGD-CIFAR100-val',
        #'PGD-CIFAR100-test',
        ]
