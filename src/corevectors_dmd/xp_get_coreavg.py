import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.get_coreVectors import get_out_activations as activations_parser
from peepholelib.coreVectors.coreVectors import CoreVectors

# Load one configuration file here
#from config_cifar100_vgg16_kernel import *
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

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    with dataset as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reduction_fns = reduction_fns,
                activations_parser = activations_parser,
                save_input = save_input,
                save_output = save_output,
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

