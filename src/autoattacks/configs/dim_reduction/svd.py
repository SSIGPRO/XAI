from configs.common import *

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD

target_layers = cfg[args.version]['layers']['linear']

# target_layers = [target_layers[-1]]

cvs_name = 'corevectors.svd'
drill_path /= 'svd'
proj_path /= 'svd'
svd_rank = 500

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

n_classifiers = {
    layer: 100 for layer in target_layers
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
                verbose = verbose
                )

        sample_in = ds._dss[f'{dataset_name}-train']['image'][0:1]

        for _l in target_layers:
                if 'fc' in _l or 'logits' in _l:
                        reducers[_l] = LinearSVD(
                                        path = proj_path,
                                        layer = _l,
                                        model = model,
                                        cv_dim = cv_dims[_l],
                                        rank = svd_rank,
                                        verbose = verbose
                                        )
                else:
                        temp_device = torch.device("cpu")
                        original_device = model.device
                        _norm = model._normalizer

                        try:
                                model.device = temp_device
                                model._model = model._model.to(temp_device)
                                if hasattr(_norm, 'mean'):
                                        _norm.mean = _norm.mean.to(temp_device)
                                        _norm.std  = _norm.std.to(temp_device)

                                reducers[_l] = Conv2dToeplitzSVD(
                                                path = proj_path,
                                                layer = _l,
                                                model = model,
                                                cv_dim = cv_dims[_l],
                                                rank = svd_rank,
                                                sample_in = sample_in.to(temp_device),
                                                device = temp_device,
                                                verbose = verbose
                                                )
                        finally:
                                model.device = original_device
                                model._model = model._model.to(original_device)
                                if hasattr(_norm, 'mean'):
                                        _norm.mean = _norm.mean.to(original_device)
                                        _norm.std  = _norm.std.to(original_device)
                                sample_in = sample_in.to(original_device)
                                reducers[_l] = Conv2dToeplitzSVD(
                                                path = proj_path,
                                                layer = _l,
                                                model = model,
                                                cv_dim = cv_dims[_l],
                                                rank = svd_rank,
                                                sample_in = sample_in,
                                                device = original_device,
                                                verbose = verbose
                                                )
