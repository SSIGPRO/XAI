import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time

# Our stuff
# model
from peepholelib.models.model_wrap import ModelWrap 

# dataset
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.coreVectors.coreVectors import CoreVectors

# corevecs
from peepholelib.coreVectors.dimReduction.vit_cls_token import ViTCLSToken 
from peepholelib.coreVectors.get_coreVectors import get_out_activations

# peepholes
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD
from peepholelib.peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vit_b_16 
from cuda_selector import auto_cuda

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cifar_path = '/srv/newpenny/dataset/CIFAR100'
    ds_path = Path.cwd()/'../../data/datasets'

    # model parameters
    bs = 64 
    n_threads = 1 

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    
    cvs_path = Path.cwd()/'../../data/corevectors_vit'
    cvs_name = 'corevectors_avg'

    drill_path = Path.cwd()/'../../data/drillers_vit'
    drill_name = 'DMD'

    phs_path = Path.cwd()/'../../data/peepholes_vit'
    phs_name = 'peepholes_avg'

    verbose = True 

    # Peepholelib
    target_layers = [
            'encoder.layers.encoder_layer_0.mlp.0',
            'encoder.layers.encoder_layer_0.mlp.3',
            ]

    loaders = [
            'CIFAR100-train',
            'CIFAR100-val',
            'CIFAR100-test' 
            ]

    #--------------------------------
    # Model 
    #--------------------------------

    nn = vit_b_16()
    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 
    model = ModelWrap(
            model=nn,
            target_modules=target_layers,
            device=device
            )

    model.update_output(
            output_layer = 'heads.head', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )

    #--------------------------------
    # Dataset 
    #--------------------------------

    # Assuming we have a parsed dataset in ds_path
    datasets = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # for each layer we define the function used to perform dimensionality reduction
    reducers = {
            'encoder.layers.encoder_layer_0.mlp.0': ViTCLSToken(
                model = model,
                layer = 'encoder.layers.encoder_layer_0.mlp.0'
                ),
            'encoder.layers.encoder_layer_0.mlp.3': ViTCLSToken(
                model = model,
                layer = 'encoder.layers.encoder_layer_0.mlp.0'
                ),
            }
    
    with datasets as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.get_coreVectors(
                datasets = ds,
                reducers = reducers,
                activations_parser = get_out_activations,
                save_input = False,
                save_output = True,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

    #--------------------------------
    # Peepholes
    #--------------------------------

    # number of channels in a conv layer. Get numbers from `nn`
    feature_sizes = {
            'encoder.layers.encoder_layer_0.mlp.0': 3072, 
            'encoder.layers.encoder_layer_0.mlp.3': 768
            }
    
    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = DMD(
                path = drill_path,
                name = f'{drill_name}.{peep_layer}',
                target_module = peep_layer,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                model = model,
                magnitude = 0.004,
                reducer = reducers[peep_layer],
                std_transform = [0.229, 0.224, 0.225],
                device = device,
                )
        
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )
        
    # fitting classifiers
    with datasets as ds, corevecs as cv:
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders,
                verbose = True
                ) 
        
        for drill_key, driller in drillers.items():
            if (drill_path/'precision.pt').exists():
                print(f'Loading DMD for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting DMD for {drill_key} time = ', time()-t0)
                driller.fit(
                        dataset = ds, 
                        corevectors = cv, 
                        loader = 'CIFAR100-train',
                        verbose=verbose
                        )
            
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with datasets as ds, corevecs as cv, peepholes as ph:
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders,
                verbose = True
                ) 

        ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose,
                )
