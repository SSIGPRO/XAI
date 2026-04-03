from configs.common import *

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.models.model_wrap import get_in_activations, get_out_activations
from peepholelib.coreVectors.dimReduction.randomProjection import RandomProjection as Reducer

import argparse
print('Loading dim_reduction common configuration...')

_parser = argparse.ArgumentParser()
_parser.add_argument('-l', '--layer', choices=['linear', 'batchnorm', 'nonlinear'], default='linear')
_parser.add_argument('-s', '--seed', type=int, default=42)
_parser.add_argument('-i', '--input', type=int, default=False)
_dim_args, _ = _parser.parse_known_args()

cvs_name = 'corevectors.random'
drill_path /= 'random'
proj_path /= 'random'
rank = 500
save_input = _dim_args.input
save_output = not _dim_args.input

if save_input: activation_parser = get_in_activations
else: activation_parser = get_out_activations

target_layers = [cfg[args.version]['layers'][_dim_args.layer][-1]]

inference_names = {
        f'{dataset_name}-train': [f'{args.model}-{args.version}'],
        f'{dataset_name}-val': [f'{args.model}-{args.version}'],
        f'{dataset_name}-test': [f'{args.model}-{args.version}']
        }

n_classifiers = {
    layer: 100 for layer in target_layers
    }

cv_dims = {
    layer: 100 for layer in target_layers
    }

model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )

dataset = ParsedDataset(
            path = ds_path,
            )

corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )

reducers = {}

with dataset as ds:
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        sample_in = ds._dss[f'{dataset_name}-train']['image'][0]

        for _l in target_layers:
                reducers[_l] = Reducer(
                        path = proj_path,
                        layer = _l,
                        model = model,
                        sample_in = sample_in,
                        seed = _dim_args.seed,
                        save_input = save_input,
                        save_output = save_output,
                        rank = rank,
                        cv_dim = cv_dims[_l],
                        verbose = verbose
                        )
