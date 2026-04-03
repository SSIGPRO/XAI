from pathlib import Path as Path
import argparse
print('Loading common configuration...')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['WRN28','WRN70'], default='WRN28')
parser.add_argument('-v', '--version', choices=['standard', 'robust'], default='standard')
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/TPAMI/review').as_posix()) # Path.cwd()/'../../../data/tests
args, remaining_argv = parser.parse_known_args()

if args.model == 'WRN28':
    from configs.models.wrn28_10 import *
elif args.model == 'WRN70':
    from configs.models.wrn70_16 import *
else:
    raise RuntimeError('Select a model <WRN28|WRN70>\'')

if args.version == 'standard':
    model = cfg['standard']['model']
    target_layers = cfg['standard']['target_layers']
elif args.version == 'robust':
    model = cfg['robust']['model']
    target_layers = cfg['robust']['target_layers']
else:
    raise RuntimeError('Select a version <standard|robust>\'')

transforms = {
            k: TransformWrap(transform=transform, input_key='image') for k in loaders 
            }

# TODO: restruct this to have svds, corevector, .. etc at leaf folders
ds_path = Path(args.path)/'datasets'/f'{dataset_name}'

svds_path = Path(args.path)/f'{dataset_name}_{args.model}'/'svds'

cvs_path = Path(args.path)/f'{dataset_name}_{args.model}'/'corevectors'

drill_path = Path(args.path)/f'{dataset_name}_{args.model}'/'drillers'

phs_path = Path(args.path)/f'{dataset_name}_{args.model}'/'peepholes'

tune_storage_path = Path(args.path)/'tuning'/f'{dataset_name}_{args.model}'

scores_file = Path(args.path)/f'{dataset_name}_{args.model}'/'temp_scores'

hyper_params_file = phs_path/f'hyperparams.pickle'

plots_path = Path(args.path)/f'{dataset_name}_{args.model}'/'temp_plots'

#--------------------------------
# Running
#--------------------------------
n_threads = 1
bs_base = 2**9
bs_atk_scale = 2**-4
tune_num_samples = 50
verbose = True
