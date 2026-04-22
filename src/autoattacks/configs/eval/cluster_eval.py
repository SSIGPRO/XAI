import sys
import argparse
from pathlib import Path as Path

from matplotlib import transforms
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.models.model_wrap import get_in_activations, get_out_activations
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD
from peepholelib.coreVectors.dimReduction.randomProjection import RandomProjection 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

print('Loading dim_eval configuration...')

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', choices=['WRN28', 'WRN70'], default='WRN28')
parser.add_argument('-r', '--reducer', choices=['svd', 'random'], default='svd')
parser.add_argument('-p', '--path', default=Path('/srv/newpenny/XAI/generated_data/attacks').as_posix())
args, remaining_argv = parser.parse_known_args()

if args.model == 'WRN28':
    from configs.models.wrn28_10 import *
elif args.model == 'WRN70':
    from configs.models.wrn70_16 import *
else:
    raise RuntimeError("Select a model <WRN28|WRN70>")

verbose = True
ds_path = Path(args.path)/'datasets'/f'{dataset_name}'

dataset = ParsedDataset(
            path = ds_path,
            )

versions = ['standard', 'robust']

reducers = {}
drillers = {}

for version in versions:

    target_layers = cfg[version]['layers']['linear']

    model = cfg[version]['model']

    cvs_name = f'corevectors.{args.reducer}'
    model_name = f'{args.model}-{version}'
    drill_path = Path(args.path)/f'{dataset_name}_{model_name}'/'drillers'/f'{args.reducer}'
    proj_path = Path(args.path)/f'{dataset_name}_{model_name}'/'dim_reduction'/f'{args.reducer}'
    phs_path = Path(args.path)/f'{dataset_name}_{model_name}'/'peepholes'

    phs_name = f'peepholes.{args.reducer}'
    drill_name = 'classifier'
    rank = 500

    inference_names = {
            f'{dataset_name}-train': [f'{model_name}'],
            f'{dataset_name}-val': [f'{model_name}'],
            f'{dataset_name}-test': 
                [
                    f'{model_name}', 
                    f'APGD-ce-{model_name}', 
                    f'APGD-t-{model_name}'
                ]
            }
    
    if args.reducer == 'svd':
        cv_dims = {
            layer: 200 for layer in target_layers
            }
        cv_dims['fc'] = 50
        cv_dims['logits'] = 50 
         
    elif args.reducer == 'random':
        cv_dims = {
        layer: 20 for layer in target_layers
        }
        save_input = input("save_input? [y/n]: ").strip().lower() == 'y'
        seed = int(input("seed: ").strip())
        save_output = not save_input
        if save_input: activation_parser = get_in_activations
        else: activation_parser = get_out_activations


    model.set_target_modules(
                target_modules = target_layers,
                verbose = verbose
                )

    reducers[version] = {}

    with dataset as ds:
            ds.load_only(
                    loaders = loaders,
                    inference_names = inference_names,
                    verbose = verbose
                    )

            sample_in = ds._dss[f'{dataset_name}-train-{args.model}-{version}']['image'][0:1]

            for _l in target_layers:
                if args.reducer == 'svd':
                    if 'fc' in _l or 'logits' in _l:
                        reducers[version][_l] = LinearSVD(
                                        path = proj_path,
                                        layer = _l,
                                        model = model,
                                        cv_dim = cv_dims[_l],
                                        rank = rank,
                                        verbose = verbose
                                        )
                    else:
                        reducers[version][_l] = Conv2dToeplitzSVD(
                                        path = proj_path,
                                        layer = _l,
                                        model = model,
                                        cv_dim = cv_dims[_l],
                                        rank = rank,
                                        sample_in = sample_in,
                                        verbose = verbose
                                        )
                elif args.reducer == 'random':
                    reducers[version][_l] = RandomProjection(
                        path = proj_path,
                        layer = _l,
                        model = model,
                        sample_in = sample_in,
                        activations_parser = activation_parser,
                        seed = seed,
                        save_input = save_input,
                        save_output = save_output,
                        rank = rank,
                        cv_dim = cv_dims[_l],
                        verbose = verbose
                        )

    n_classifiers = {
        layer: 100 for layer in target_layers
        }

    drillers[version] = {}
    for _l in target_layers:

            drillers[version][_l] = tGMM(
                    path = drill_path,
                    name = f'{drill_name}.GMM.{_l}.{n_classes}.{cv_dims[_l]}.{n_classifiers[_l]}',
                    target_module = _l,
                    nl_classifier = n_classifiers[_l],
                    nl_model = n_classes,
                    n_features = cv_dims[_l],
                    reducer = reducers[version][_l],
                    device = device
                    )
            drillers[version][_l].load()