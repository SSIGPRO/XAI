import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/Peepholes-Analysis/src').as_posix())

from peepholelib.coreVectors.dimReduction.svds.vit_linear_svd import ViTLinearSVD
from peepholelib.coreVectors.dimReduction.svds.linear_svd import LinearSVD
from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD
from peepholelib.peepholes.parsers.peepholes_parsers import dotprod_parser as peepholes_parser

from configs.common import *

cvs_name = 'corevectors'
phs_name = 'peepholes'
drill_name = 'classifier'

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
    
elif args.model == 'ViTB16':
    target_layers = [
        f'encoder.layers.encoder_layer_{i}.mlp.{j}' for i in range(12) for j in [0,3]
        ]
    target_layers.append(output_layer)

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
elif args.model == 'SwinB':
    target_layers = [
        "features.0.0",

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
else:
    raise RuntimeError(f'Model {args.model} not supported try <VGG16|ViTB16|ResNet50|SwinB>')

target_layers = [f'model.{layer}' for layer in target_layers]

model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )

# Ray Tune
num_samples = 100

# Overwrite verbose
verbose=False

min_n_classifier = 100 
max_n_classifier = 1000

min_cv_dim = 10
max_cv_dim = {}
# set a maximum value for cv_size,  for the last layer max_size = 100
for _l in target_layers:
    max_cv_dim[_l] = n_classes if any(k in _l for k in ("classifier", "heads")) else 500

red_classes = {}

for _l in target_layers:
    if 'mlp' in _l: red_classes[_l] = ViTLinearSVD  
    elif 'heads.head'  in _l or 'classifier' in _l:  red_classes[_l] = LinearSVD
    else :  red_classes[_l] = Conv2dToeplitzSVD