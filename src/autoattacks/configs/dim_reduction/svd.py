from configs.common import *

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.reducers.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.reducers.conv2d_toeplitz_svd import Conv2dToeplitzSVD

target_layers = cfg[args.version]['layers']['linear']

cvs_name = 'corevectors.svd'
drill_path /= 'svd'
proj_path /= 'svd'
svd_rank = 500

inference_names = {
        f'{dataset_name}-train': [f'{args.model}-{args.version}'],
        f'{dataset_name}-val': [f'{args.model}-{args.version}'],
        f'{dataset_name}-test': [f'{args.model}-{args.version}']
        }

n_classifiers = {
    layer: 50 for layer in target_layers
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
                if 'fc' in _l:
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

                        try:
                                model.device = temp_device
                                model._model = model._model.to(temp_device)

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
