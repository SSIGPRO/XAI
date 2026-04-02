from pathlib import Path as Path
import argparse
print('Loading common configuration...')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['VGG16','ViTB16', 'ResNet50', 'SwinB'], default='VGG16')
parser.add_argument('-d', '--dataset', choices=['CIFAR100', 'ImageNet'], default='CIFAR100')
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/TPAMI/review').as_posix()) # Path.cwd()/'../../data/tests'
args, remaining_argv = parser.parse_known_args()

# import configs
if args.dataset == 'CIFAR100':
    from configs.datasets.cifar import *
elif args.dataset == 'ImageNet':
    from configs.datasets.imagenet import *
else:
    raise RuntimeError('Select a dataset <CIFAR100|ImageNet>\'')

if args.model == 'VGG16':
    from configs.models.vgg import *
elif args.model == 'ViTB16':
    from configs.models.vit import *
elif args.model == 'ResNet50':
    from configs.models.resnet import *
elif args.model == 'SwinB':
    from configs.models.swin import *
else:
    raise RuntimeError('Select a model <VGG16|ViTB16|ResNet50|SwinB>\'')

transforms = {
            k: TransformWrap(transform=transform, input_key='image') for k in loaders 
            }

model = cfg[args.dataset]["model"]
n_classes = cfg[args.dataset]["n_classes"]

# TODO: restruct this to have svds, corevector, .. etc at leaf folders
ds_path = Path(args.path)/'datasets'/f'{args.dataset}'

svds_path = Path(args.path)/f'{args.dataset}_{args.model}'/'svds'

cvs_path = Path(args.path)/f'{args.dataset}_{args.model}'/'corevectors'

drill_path = Path(args.path)/f'{args.dataset}_{args.model}'/'drillers'

phs_path = Path(args.path)/f'{args.dataset}_{args.model}'/'peepholes'

tune_storage_path = Path(args.path)/'tuning'/f'{args.dataset}_{args.model}'

scores_file = Path(args.path)/f'{args.dataset}_{args.model}'/'temp_scores'

hyper_params_file = phs_path/f'hyperparams.pickle'

plots_path = Path(args.path)/f'{args.dataset}_{args.model}'/'temp_plots'

#--------------------------------
# Running
#--------------------------------
n_threads = 1
bs_base = 2**9
bs_atk_scale = 2**-4
tune_num_samples = 50
