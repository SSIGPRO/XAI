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
from peepholelib.peepholes.FeatureMixing.tgmm import GMM as tGMM
from peepholelib.peepholes.parsers.peepholes_parsers import dotprod_parser, mahalanobis_parser

from configs.common import *

datasets = ParsedDataset(
            path = ds_path,
            )

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
    
elif args.model == 'ViTB16':
    target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
        ]
    target_layers.append(output_layer)

    target_layers = [f'model.{layer}' for layer in target_layers]
