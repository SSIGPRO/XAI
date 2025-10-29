import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
import peepholelib
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.parsers import get_images  
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD 
from peepholelib.peepholes.peepholes import Peepholes

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":
    #--------------------------------
    # Dataset 
    #--------------------------------
                                      
    dataset = ParsedDataset(
            path = ds_path,
            )

    #--------------------------------
    # Model 
    #--------------------------------
    
    model = ModelWrap(
            model = Model(),
            device = device
            )
    
    model.update_output(
            output_layer = output_layer, 
            to_n_classes = n_classes,
            overwrite = True 
            )
    
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )


    corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                )

    #--------------------------------
    # Peepholes 
    #--------------------------------
    with dataset as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        cv.load_only(
                loaders = loaders, 
                verbose = verbose 
                ) 

        drillers = {}
        for _layer in target_layers:
            drillers[_layer] = DMD(
                    path = drill_path,
                    name = drill_name+'.'+_layer,
                    nl_model = n_classes,
                    n_features = feature_sizes_dmd[_layer],
                    parser = get_images,
                    model = model,
                    layer = _layer,
                    magnitude = magnitude,
                    std_transform = [0.300, 0.287, 0.294],
                    device = device,
                    parser_act = parser_act
                    )

            if (drill_path/drillers[_layer]._suffix/'precision.pt').exists():
                print(f'Loading DMD for {_layer}') 
                drillers[_layer].load()
            else:
                drillers[_layer].fit(
                        dataset = ds,
                        corevectors = cv,
                        loader = 'CIFAR100-train',
                        drill_key = _layer,
                        verbose=verbose
                        )
            
                # save classifiers
                print(f'Saving classifier for {_layer}')
                drillers[_layer].save()

        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )

        with peepholes as ph:
            ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = 1,
                verbose = verbose 
                )
            
