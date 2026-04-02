import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/Peepholes-Analysis/src').as_posix())

# Peephoelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset  
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.coreVectors.dimReduction.svds.vit_linear_svd import ViTLinearSVD
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD
from peepholelib.peepholes.ClassDependentClassifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.parsers.peepholes_parsers import dotprod_parser as peepholes_parser
from peepholelib.datasets.parsedDataset import ParsedDataset  
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds.vit_linear_svd import ViTLinearSVD 
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD

from configs.common import *

cvs_name = 'corevectors'
phs_name = 'peepholes_cdc'
drill_name = 'CDclassifier'

if args.model == 'VGG16':
    target_layers = [
        'features.7',
        'features.10',
        'features.12',
        'features.14',
        'features.17',
        'features.19',
        'features.21',
        'features.24',
        'features.26',
        'features.28',
        'classifier.0',
        'classifier.3',
        'classifier.6',
        ]
    
    target_layers = [f'model.{layer}' for layer in target_layers]

    cv_dims = {_l: 100 for _l in target_layers} 

    cv_dims[target_layers[-1]] = 50

    n_classifiers = {_l: 3 for _l in target_layers}

elif args.model == 'ViTB16':
    target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
        ]
    target_layers.append(output_layer)

    target_layers = [f'model.{layer}' for layer in target_layers]

    cv_dims = {_l: 100 for _l in target_layers} 

    cv_dims[target_layers[-1]] = 50

    n_classifiers = {_l: 3 for _l in target_layers}

elif args.model == 'ResNet50':
    target_layers = [
        'conv1',
        'layer1.0.conv1',
        'layer1.0.conv2',
        'layer1.0.conv3',
        'layer1.1.conv1',
        'layer1.1.conv2',
        'layer1.1.conv3',
        'layer1.2.conv1',
        'layer1.2.conv2',
        'layer1.2.conv3',

        'layer2.0.conv1',
        'layer2.0.conv2',
        'layer2.0.conv3',
        'layer2.1.conv1',
        'layer2.1.conv2',
        'layer2.1.conv3',
        'layer2.2.conv1',
        'layer2.2.conv2',
        'layer2.2.conv3',
        'layer2.3.conv1',
        'layer2.3.conv2',
        'layer2.3.conv3',

        'layer3.0.conv1',
        'layer3.0.conv2',
        'layer3.0.conv3',
        'layer3.1.conv1',
        'layer3.1.conv2',
        'layer3.1.conv3',
        'layer3.2.conv1',
        'layer3.2.conv2',
        'layer3.2.conv3',
        'layer3.3.conv1',
        'layer3.3.conv2',
        'layer3.3.conv3',
        'layer3.4.conv1',
        'layer3.4.conv2',
        'layer3.4.conv3',
        'layer3.5.conv1',
        'layer3.5.conv2',
        'layer3.5.conv3',

        'layer4.0.conv1',
        'layer4.0.conv2',
        'layer4.0.conv3',
        'layer4.1.conv1',
        'layer4.1.conv2',
        'layer4.1.conv3',
        'layer4.2.conv1',
        'layer4.2.conv2',
        'layer4.2.conv3',
        'fc',
        ]
    target_layers = [f'model.{layer}' for layer in target_layers]

    cv_dims = {_l: 100 for _l in target_layers} 

    cv_dims[target_layers[-1]] = 50

    n_classifiers = {_l: 3 for _l in target_layers}
    

elif args.model == 'SwinB':
    target_layers = [
        #"features.0.0",

        # "features.1.0.attn.qkv",
        # "features.1.0.attn.proj",
        "features.1.0.mlp.0", 
        "features.1.0.mlp.3",
        # "features.1.1.attn.qkv",
        # "features.1.1.attn.proj",
        "features.1.1.mlp.0",
        "features.1.1.mlp.3",

        # "features.3.0.attn.qkv",
        # "features.3.0.attn.proj",
        "features.3.0.mlp.0", 
        "features.3.0.mlp.3",
        # "features.3.1.attn.qkv",
        # "features.3.1.attn.proj",
        "features.3.1.mlp.0", 
        "features.3.1.mlp.3",

        # "features.4.reduction",

        "features.5.0.mlp.0", 
        "features.5.0.mlp.3",
        # "features.5.1.attn.proj",
        "features.5.1.mlp.0", 
        "features.5.1.mlp.3",
        # "features.5.2.attn.qkv",
        # "features.5.2.attn.proj",
        "features.5.2.mlp.0", 
        "features.5.2.mlp.3",
        # "features.5.3.attn.qkv",
        # "features.5.3.attn.proj",
        "features.5.3.mlp.0", 
        "features.5.3.mlp.3",
        # "features.5.4.attn.qkv",
        # "features.5.4.attn.proj",
        "features.5.4.mlp.0", 
        "features.5.4.mlp.3",
        # "features.5.5.attn.qkv",
        # "features.5.5.attn.proj",
        "features.5.5.mlp.0", 
        "features.5.5.mlp.3",
        # "features.5.6.attn.qkv",
        # "features.5.6.attn.proj",
        "features.5.6.mlp.0", 
        "features.5.6.mlp.3",
        # "features.5.7.attn.qkv",
        # "features.5.7.attn.proj",
        "features.5.7.mlp.0", 
        "features.5.7.mlp.3",
        # "features.5.8.attn.qkv",
        # "features.5.8.attn.proj",
        "features.5.8.mlp.0", 
        "features.5.8.mlp.3",
        # "features.5.9.attn.qkv",
        # "features.5.9.attn.proj",
        "features.5.9.mlp.0", 
        "features.5.9.mlp.3",
        # "features.5.10.attn.qkv",
        # "features.5.10.attn.proj",
        "features.5.10.mlp.0", 
        "features.5.10.mlp.3",
        # "features.5.11.attn.qkv",
        # "features.5.11.attn.proj",
        "features.5.11.mlp.0", 
        "features.5.11.mlp.3",
        # "features.5.12.attn.qkv",
        # "features.5.12.attn.proj",
        "features.5.12.mlp.0", 
        "features.5.12.mlp.3",
        # "features.5.13.attn.qkv",
        # "features.5.13.attn.proj",
        "features.5.13.mlp.0", 
        "features.5.13.mlp.3",
        # "features.5.14.attn.qkv",
        # "features.5.14.attn.proj",
        "features.5.14.mlp.0", 
        "features.5.14.mlp.3",
        # "features.5.15.attn.qkv",
        # "features.5.15.attn.proj",
        "features.5.15.mlp.0", 
        "features.5.15.mlp.3",
        # "features.5.16.attn.qkv",
        # "features.5.16.attn.proj",
        "features.5.16.mlp.0", 
        "features.5.16.mlp.3",
        # "features.5.17.attn.qkv",
        # "features.5.17.attn.proj",
        "features.5.17.mlp.0", 
        "features.5.17.mlp.3",

        # "features.6.reduction",

        # "features.7.0.attn.qkv",
        # "features.7.0.attn.proj",
        "features.7.0.mlp.0", 
        "features.7.0.mlp.3",
        # "features.7.1.attn.qkv",
        # "features.7.1.attn.proj",
        "features.7.1.mlp.0", 
        "features.7.1.mlp.3",
        "head",
        ]
    target_layers = [f'model.{layer}' for layer in target_layers]

    cv_dims = {_l: 100 for _l in target_layers} 

    cv_dims[target_layers[-1]] = 50

    n_classifiers = {_l: 3 for _l in target_layers}

model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )

#--------------------------------
# SVDs 
#--------------------------------

svd_rank = 500 

dataset = ParsedDataset(
            path = ds_path,
            )

#--------------------------------
# Corevectors 
#--------------------------------
corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )

with dataset as ds: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        sample_in = ds._dss[f'{args.dataset}-train']['image'][0]

        svds = {}
        for _l in target_layers:
                if 'head'  in _l or 'classifier' in _l or 'fc' in _l:
                        svds[_l] = LinearSVD(
                                        path = svds_path,
                                        layer = _l,
                                        model = model,
                                        cv_dim = cv_dims[_l],
                                        rank = svd_rank,
                                        verbose = verbose
                                        )
                        
                elif 'mlp' in _l:

                        if 'encoder' in _l:
                                token_reduction = 'first'
                        elif 'features' in _l:
                                token_reduction = 'mean'
                        
                        svds[_l] = ViTLinearSVD(
                                        path = svds_path,
                                        layer = _l,
                                        model = model,
                                        cv_dim = cv_dims[_l],
                                        token_reduction= token_reduction,
                                        rank = svd_rank,
                                        verbose = verbose
                                        )
                else:
                        temp_device = torch.device("cpu")
                        original_device = model.device

                        try:
                                model.device = temp_device
                                model._model = model._model.to(temp_device)

                                svds[_l] = Conv2dToeplitzSVD(
                                                path = svds_path,
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
                                svds[_l] = Conv2dToeplitzSVD(
                                                path = svds_path,
                                                layer = _l,
                                                model = model,
                                                cv_dim = cv_dims[_l],
                                                rank = svd_rank,
                                                sample_in = sample_in.to(temp_device),
                                                device = original_device,
                                                verbose = verbose
                                                )
              

#--------------------------------
# Peepholes 
#--------------------------------

peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )
                    
drillers = {}
# cls_kwargs = {'covariance_type' = diag}
for _l in target_layers:

        drillers[_l] = tGMM(
                path = drill_path,
                name = f'{drill_name}.GMM.{_l}.{n_classes}.{cv_dims[_l]}.{n_classifiers[_l]}',
                target_module = _l,
                nl_classifier = n_classifiers[_l],
                nl_model = n_classes,
                n_features = cv_dims[_l],
                reducer = svds[_l],
                device = device
                )
        
        