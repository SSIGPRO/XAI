from pathlib import Path as Path
import argparse
print('Loading common configuration...')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',  choices=['WRN28','WRN70'], default='WRN28')
parser.add_argument('-v', '--version', choices=['standard', 'robust'], default='standard')
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
args, remaining_argv = parser.parse_known_args()

if args.model == 'WRN28':
    from configs.models.wrn28_10 import *
elif args.model == 'WRN70':
    from configs.models.wrn70_16 import *
else:
    raise RuntimeError('Select a model <WRN28|WRN70>\'')

if args.version == 'standard':
    model = cfg['standard']['model']
elif args.version == 'robust':
    model = cfg['robust']['model']
else:
    raise RuntimeError('Select a version <standard|robust>\'')

transforms = {
            k: TransformWrap(transform=transform, input_key='image') for k in loaders 
            }

model_name = f'{args.model}-{args.version}'
# TODO: restruct this to have svds, corevector, .. etc at leaf folders
ds_path = Path(args.path)/'datasets_'/f'{dataset_name}'

proj_path = Path(args.path)/f'{dataset_name}_{model_name}_'/'dim_reduction'

cvs_path = Path(args.path)/f'{dataset_name}_{model_name}_'/'corevectors'

drill_path = Path(args.path)/f'{dataset_name}_{model_name}_'/'drillers'

phs_path = Path(args.path)/f'{dataset_name}_{model_name}_'/'peepholes'

tune_storage_path = Path(args.path)/'tuning'/f'{dataset_name}_{model_name}_'

scores_file = Path(args.path)/f'{dataset_name}_{model_name}_'/'temp_scores'

hyper_params_file = phs_path/f'hyperparams.pickle'

plots_path = Path(args.path)/f'{dataset_name}_{model_name}_'/'temp_plots'

#--------------------------------
# Running
#--------------------------------
n_threads = 1
bs_base = 2**9
bs_atk_scale = 2**-4
tune_num_samples = 50
verbose = True
