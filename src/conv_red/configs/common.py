from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['VGG','MobileNet', 'ResNet'], default='VGG')
parser.add_argument('-r', '--reduction', choices=['avgpooling', 'toeplitz', 'kernel'], default='kernel')
parser.add_argument('-a', '--analysis', choices=['MACS', 'DMD'], default='MACS')
parser.add_argument('-d', '--data_path', default=Path.cwd()/'../../data')
args = parser.parse_args()

# import configs
if args.model == 'VGG':
    from configs.vgg import *
elif args.model == 'MobileNet':
    from configs.mobilenet import *
elif args.model == 'ResNet':
    from configs.resnet import *

if args.reduction == 'avgpooling':
    from configs.avgpooling import *
elif args.reduction == 'toeplitz':
    from configs.toeplitz import *
elif args.reduction == 'kernel':
    from configs.kernel import *

if args.analysis == 'MACS':
    from configs.macs import *
elif args.analysis == 'DMD':
    from configs.dmd import *

from peepholelib.datasets.functional.transforms import TransformWrap 
from peepholelib.datasets.functional.transforms import means as _means, stds as _stds 
normalization_mean = _means['CIFAR100']
normalization_std = _stds['CIFAR100']

#--------------------------------
# Paths and Definitions 
#--------------------------------
cifar_path = '/srv/newpenny/dataset/CIFAR100'
cifarc_path = '/srv/newpenny/dataset/CIFAR-100-C'
svhn_path = '/srv/newpenny/dataset/SVHN' 
places_path = '/srv/newpenny/dataset/Places365'

# TODO: restruct this to have svds, corevector, .. etc at leaf folders
ds_path = Path(args.data_path)/'datasets'

svds_path = Path(args.data_path)/args.model/'svds'/args.reduction

cvs_path = Path(args.data_path)/args.model/'corevectors'/args.reduction
cvs_name = 'cvs' 

drill_path = Path(args.data_path)/args.model/'drillers'/args.reduction/args.analysis
drill_name = 'driller' 

phs_path = Path(args.data_path)/args.model/'peepholes'/args.reduction/args.analysis
phs_name = 'phs' 

tune_storage_path = Path(args.data_path)/args.model/'tuning'/args.reduction/args.analysis

hyper_params_file = phs_path/f'hyperparams.pickle'

plots_path = Path.cwd()/'temp_plots'

#--------------------------------
# Runing
#--------------------------------
seed = 2
n_threads = 1
verbose = True 
n_classes = 100
bs_base = 2**10
bs_atk_scale = 2**-4
tune_num_samples = 50 

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
        ]

transforms = {
        k: TransformWrap(transform=transform, input_key='image') for k in loaders 
        }

inference_names = {
        'CIFAR100-train': [args.model],     
        'CIFAR100-val': [args.model, 'BIM-'+args.model, 'CW-'+args.model],
        'CIFAR100-test': [args.model, 'BIM-'+args.model, 'CW-'+args.model],
        'CIFAR100-C-val-c4': [args.model],
        'CIFAR100-C-test-c4': [args.model],
        'SVHN-val': [args.model],
        'SVHN-test': [args.model],
        'Places365-val': [args.model],
        'Places365-test': [args.model],
        }