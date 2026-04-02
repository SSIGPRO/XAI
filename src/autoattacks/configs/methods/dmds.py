import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/Peepholes-Analysis/src').as_posix())

# Peephoelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset  
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD 
from peepholelib.coreVectors.dimReduction.avgPooling import AvgPooling
from peepholelib.coreVectors.dimReduction.vit_cls_token import ViTCLSToken
from peepholelib.coreVectors.dimReduction.vit_token_wise_mean import ViTTokenWiseMean
from peepholelib.models.model_wrap import get_out_activations

from configs.common import *

inference_names = {
        f'{args.dataset}-train': [args.model],     
        f'{args.dataset}-val': [args.model, 'BIM-'+args.model],#, 'CW-'+args.model],
        f'{args.dataset}-test': [args.model, 'BIM-'+args.model],#, 'CW-'+args.model],
        f'{args.dataset}-C-val-c4': [args.model],
        f'{args.dataset}-C-test-c4': [args.model],
        f'{args.dataset}-SVHN-val': [args.model],
        'SVHN-test': [args.model],
        'Places365-val': [args.model],
        'Places365-test': [args.model],
        }

cvs_name = 'coreavgs'
phs_name = 'peepavgs'
drill_name = 'DMD'
magnitude = 0.004

if args.model == 'VGG16':
    target_layers = [
           'features.2',
           'features.7',
           'features.14',
           'features.21',
           'features.28'
            ]  
    Reducer = AvgPooling
    feature_sizes_dmd = {
        'features.2': 64, 
        'features.7': 128, 
        'features.14': 256, 
        'features.21': 512,
        'features.28': 512,
    }

elif args.model == 'ViTB16':
    target_layers = [f'encoder.layers.encoder_layer_{i}.mlp.3' for i in range(12)]
    Reducer = ViTCLSToken
    feature_sizes_dmd = { f'encoder.layers.encoder_layer_{i}.mlp.3': 768 for i in range(12)}

elif args.model == 'ResNet50':
    target_layers = [
        'layer1.0.conv3',
        'layer1.1.conv3',
        'layer1.2.conv3',
        'layer2.0.conv3',
        'layer2.1.conv3',
        'layer2.2.conv3',
        'layer2.3.conv3',
        'layer3.0.conv3',
        'layer3.1.conv3',
        'layer3.2.conv3',
        'layer3.3.conv3',
        'layer3.4.conv3',
        'layer3.5.conv3',
        'layer4.0.conv3',
        'layer4.1.conv3',
        'layer4.2.conv3',
        ]
    Reducer = AvgPooling
    feature_sizes_dmd = {
        'layer1.0.conv3': 256,
        'layer1.1.conv3': 256,
        'layer1.2.conv3': 256,
        'layer2.0.conv3': 512,
        'layer2.1.conv3': 512,
        'layer2.2.conv3': 512,
        'layer2.3.conv3': 512,
        'layer3.0.conv3': 1024,
        'layer3.1.conv3': 1024,
        'layer3.2.conv3': 1024,
        'layer3.3.conv3': 1024,
        'layer3.4.conv3': 1024,
        'layer3.5.conv3': 1024,
        'layer4.0.conv3': 2048,
        'layer4.1.conv3': 2048,
        'layer4.2.conv3': 2048,
    }

elif args.model == 'SwinB':
    submodules = {
        1: [0,1],
        3: [0,1],
        5: range(0,18),
        7: [0,1]
        }
    target_layers = [
        f'features.{sub}.{p}.mlp.3' 
        for sub in submodules 
        for p in submodules[sub]
        ]
    Reducer = ViTTokenWiseMean
    feature_sizes_dmd = { 
        "features.1.0.mlp.3": 128,      
        "features.1.1.mlp.3": 128,
        "features.3.0.mlp.3": 256,        
        "features.3.1.mlp.3": 256,
        "features.5.0.mlp.3": 512,         
        "features.5.1.mlp.3": 512,
        "features.5.2.mlp.3": 512,
        "features.5.3.mlp.3": 512, 
        "features.5.4.mlp.3": 512, 
        "features.5.5.mlp.3": 512,
        "features.5.6.mlp.3": 512,
        "features.5.7.mlp.3": 512,    
        "features.5.8.mlp.3": 512,    
        "features.5.9.mlp.3": 512,    
        "features.5.10.mlp.3": 512,    
        "features.5.11.mlp.3": 512,     
        "features.5.12.mlp.3": 512,    
        "features.5.13.mlp.3": 512,     
        "features.5.14.mlp.3": 512,     
        "features.5.15.mlp.3": 512,     
        "features.5.16.mlp.3": 512,    
        "features.5.17.mlp.3": 512,
        "features.7.0.mlp.3": 1024,    
        "features.7.1.mlp.3": 1024,        
    }

else: 
    raise NotImplementedError(f'Model {args.model} not supported')

model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )

dataset = ParsedDataset(
            path = ds_path,
            )

#--------------------------------
# Corevectors 
#--------------------------------
coreavgs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )

reducers = {}
drillers = {}

for _layer in target_layers:
        reducers[_layer] = Reducer(
                model = model,
                layer = _layer
                )
            
        drillers[_layer] = DMD(
            path = drill_path,
            name = f'{drill_name}.{_layer}',
            target_module = _layer,
            nl_model = n_classes,
            n_features = feature_sizes_dmd[_layer],
            model = model,
            magnitude = magnitude,
            reducer = reducers[_layer],
            std_transform = [0.229, 0.224, 0.225],
            device = device,
            )