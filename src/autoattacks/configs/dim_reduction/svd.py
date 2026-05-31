from configs.common import *
import argparse
import torch

# Peepholelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.vit_linear_svd import ViTLinearSVD 
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD
from corevectors.token_wise_linear_svd import TokenWiseLinearSVD

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
        '--token-wise',
        action='store_true',
        help='For no-CLS transformer token layers, save one SVD projection per token instead of mean-reducing tokens.',
        )
_dim_args, _ = _parser.parse_known_args()

_cfg_layers = cfg[args.version]['layers']
if isinstance(_cfg_layers, dict):
        target_layers = _cfg_layers['linear']
else:
        target_layers = _cfg_layers

reduction_name = 'token_svd' if _dim_args.token_wise else 'svd'
cvs_name = f'corevectors.{reduction_name}'
drill_path /= reduction_name
proj_path /= reduction_name
svd_rank = 500
corevector_path = cvs_path / reduction_name if _dim_args.token_wise else cvs_path
corevector_names = None

NO_CLS_TOKEN_MODELS = {'XCiT', 'Swin'}


def is_classifier_layer(layer):
        return 'head' in layer or layer in {'fc', 'logits'} or layer.endswith('.logits')


def is_no_cls_token_layer(layer):
        if args.model not in NO_CLS_TOKEN_MODELS:
                return False
        return 'cls' not in layer and 'encoder' not in layer


def is_token_linear_layer(layer):
        if 'mlp' in layer:
                return True
        if args.model not in NO_CLS_TOKEN_MODELS:
                return False
        return isinstance(model._target_modules[layer], torch.nn.Linear) and not is_classifier_layer(layer)


def build_linear_svd(layer):
        return LinearSVD(
                        path = proj_path,
                        layer = layer,
                        model = model,
                        cv_dim = cv_dims[layer],
                        rank = svd_rank,
                        verbose = verbose
                        )


class CpuConv2dToeplitzSVD(Conv2dToeplitzSVD):
        def __call__(self, **kwargs):
                act_data = kwargs['act_data']
                return super().__call__(act_data=act_data.to(self.reduct_m.device))

inference_names = {
        f'{dataset_name}-train': [f'{model_name}'],
        f'{dataset_name}-val': [f'{model_name}'],
        f'{dataset_name}-test': 
            [
                f'{model_name}',
                f'APGD-ce-{model_name}',
                f'APGD-t-{model_name}',
                f'FAB-t-{model_name}',
                f'Square-{model_name}'
            ]
        }

analysis_loaders = [
        loader for loader in loaders if loader != f'{dataset_name}-train'
        ]
analysis_inference_names = {
        loader: inference_names[loader] for loader in analysis_loaders
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
        path = corevector_path,
        name = cvs_name,
        model = model,
        )

reducers = {}

with dataset as ds:
        ds.load_only(
                loaders = analysis_loaders,
                verbose = verbose
                )

        sample_in = ds._dss[analysis_loaders[0]][0:1]['image'].squeeze(0)
        
        for _l in target_layers:
                _module = model._target_modules[_l]
                # Tokenized transformer linear layers need token-aware SVD.
                # XCiT names MLP layers as fc1/fc2, so this branch must run
                # before the classifier-head LinearSVD branch.
                if isinstance(_module, torch.nn.Linear) and is_token_linear_layer(_l):

                        if 'encoder' in _l:
                                token_reduction = 'first'
                        elif 'cls_attn' in _l:
                                # XCiT ClassAttentionBlock: cls token is prepended at position 0
                                token_reduction = 'first'
                        else:
                                # XCiT XCABlock: purely spatial tokens, no cls token
                                token_reduction = 'mean'

                        Reducer = (
                                TokenWiseLinearSVD
                                if _dim_args.token_wise and is_no_cls_token_layer(_l)
                                else ViTLinearSVD
                                )

                        reducers[_l] = Reducer(
                                        path = proj_path,
                                        layer = _l,
                                        model = model,
                                        cv_dim = cv_dims[_l],
                                        token_reduction= token_reduction,
                                        rank = svd_rank,
                                        verbose = verbose
                                        )

                elif isinstance(_module, torch.nn.Linear):
                        reducers[_l] = build_linear_svd(_l)

                elif isinstance(_module, torch.nn.Conv2d):
                        temp_device = torch.device("cpu")
                        original_device = model.device
                        _norm = model._normalizer

                        try:
                                model.device = temp_device
                                model._model = model._model.to(temp_device)
                                if hasattr(_norm, 'mean'):
                                        _norm.mean = _norm.mean.to(temp_device)
                                        _norm.std  = _norm.std.to(temp_device)

                                reducers[_l] = CpuConv2dToeplitzSVD(
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

                else:
                        raise RuntimeError(
                                        f'Unsupported target layer {_l}: '
                                        f'{_module.__class__.__name__}'
                                        )
