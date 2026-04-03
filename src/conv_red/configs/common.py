from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['vgg', 'wrn','robustwrn'], default='robustwrn')
parser.add_argument('-d', '--data_path', default=Path.cwd()/'../../data')
args = parser.parse_args()

# just vgg for the moment

if args.model == 'vgg':
    from configs.vgg import *
elif args.model == 'wrn':
    from configs.wrn import *
elif args.model == 'robustwrn':
    from configs.robustwrn import *

#--------------------------------
# Paths and Definitions 
#--------------------------------
# focusing on CIFAR100

cifar_path = '/srv/newpenny/dataset/CIFAR100'
ds_path = Path(args.data_path)/args.model/'datasets'

#--------------------------------
# Running
#--------------------------------

seed = 2
n_threads = 1
verbose = True 
n_classes = 100
bs_base = 2**8
bs_atk_scale = 2**-2 #-4 og

#--------------------------------
# Defs 
#--------------------------------

loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        ]
