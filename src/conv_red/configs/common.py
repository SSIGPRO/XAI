from pathlib import Path as Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['vgg', 'wrn'], default='wrn')
parser.add_argument('-d', '--data_path', default=Path.cwd()/'../../data')
args = parser.parse_args()

<<<<<<< HEAD
# import configs
=======
# just vgg for the moment

>>>>>>> d2de4fa (start)
if args.model == 'vgg':
    from configs.vgg import *
elif args.model == 'wrn':
    from configs.wrn import *

#--------------------------------
# Paths and Definitions 
#--------------------------------
# focusing on CIFAR100

<<<<<<< HEAD
# TODO: restruct this to have svds, corevector, .. etc at leaf folders
ds_path = Path(args.data_path)/args.model/'datasets'

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

=======
cifar_path = '/srv/newpenny/dataset/CIFAR100'
ds_path = Path(args.data_path)/args.model/'datasets'

>>>>>>> d2de4fa (start)
#--------------------------------
# Running
#--------------------------------

seed = 2
n_threads = 1
verbose = True 
n_classes = 100
<<<<<<< HEAD
bs_base = 2**10
bs_atk_scale = 2**-4
tune_num_samples = 50
=======
bs_base = 2**8
bs_atk_scale = 2**-2 #-4 og
>>>>>>> d2de4fa (start)

#--------------------------------
# Defs 
#--------------------------------

loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        ]
