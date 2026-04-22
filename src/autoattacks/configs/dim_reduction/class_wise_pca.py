from configs.common import *
import torch

from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.pca.class_wise_streaming_pca import ClassWiseStreamingPCA

target_layers = cfg[args.version]['layers']['linear']

cvs_name = 'corevectors.class_wise_pca'
drill_path /= 'class_wise_pca'
proj_path /= 'class_wise_pca'
pca_rank = 200

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

cv_dims = {
    layer: 200 for layer in target_layers
    }

if args.version == 'standard':
        cv_dims['fc'] = 50
elif args.version == 'robust':
        cv_dims['logits'] = 50
else:
        raise ValueError('Select a version <standard|robust>\'')

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
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        for _l in target_layers:
                reducers[_l] = ClassWiseStreamingPCA(
                        path = proj_path,
                        layer = _l,
                        model = model,
                        dataset = ds,
                        ds_key = f'{dataset_name}-train-{model_name}',
                        cv_dim = cv_dims[_l],
                        rank = pca_rank,
                        n_classes = n_classes,
                        batch_size = bs_base,
                        n_threads = n_threads,
                        verbose = verbose
                        )
