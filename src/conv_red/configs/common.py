from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['vgg','mobile_net'], default='vgg')
parser.add_argument('-r', '--reduction', choices=['maxp', 'toeplitz', 'kernel'], default='kernel')
parser.add_argument('-a', '--analysis', choices=['macs', 'dmd'], default='macs')
args = parser.parse_args()

# import configs
from configs.common import *

if args.model == 'vgg':
    from configs.vgg import *
elif args.model == 'mobile_net':
    from configs.vit import *

if args.reduction == 'maxp':
    from configs.maxp import *
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

n_classes = 100

#--------------------------------
# Runing
#--------------------------------
seed = 2
n_threads = 1
    
verbose = True 
