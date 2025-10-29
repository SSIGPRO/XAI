import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap, means, stds 
from peepholelib.coreVectors.coreVectors import CoreVectors

import matplotlib.pyplot as plt

# Load one configuration file here
#from config_cifar100_vgg16_kernel import *
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
elif sys.argv[1] == 'vgg_imagenet':
    from config_imagenet_vgg16 import *
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
            model = model,
            device = device
            )

#     model.update_output(
#             output_layer = output_layer, 
#             to_n_classes = n_classes,
#             overwrite = True 
#             )

#     model.load_checkpoint(
#             name = model_name,
#             path = model_dir,
#             verbose = verbose
#             )

    model.normalize_model(mean=means[dataset_name], std=stds[dataset_name])

    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    with dataset as ds: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )

        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = ds._dss['ImageNet-train']['image'][0],
                svd_fns = svd_fns,
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
        
        with corevecs as cv: 
            # add svd to reduction_fns
            for _layer in reduction_fns:
                if 'features' in _layer:
                    reduction_fns[_layer].keywords['layer'] = model._target_modules[_layer]
                reduction_fns[_layer].keywords['svd'] = model._svds[_layer] 
            
            # computing the corevectors
            cv.get_coreVectors(
                    datasets = ds,
                    reduction_fns = reduction_fns,
                    save_input = save_input,
                    save_output = save_output,
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose = verbose
                    )

            if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
                cv.normalize_corevectors(
                        wrt = 'ImageNet-train',
                        to_file = cvs_path/(cvs_name+'.normalization.pt'),
                        #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                        #loaders = [
                        #    f'{portion}-ood-{ood}' for portion in ['test', 'val'] for ood in ds_kwargs['ood_dss']],
                        #    [
                        #        'test-ood-c0', 'test-ood-c1', 'test-ood-c2', 'test-ood-c3', 'test-ood-c4',
                        #        'val-ood-c0', 'val-ood-c1', 'val-ood-c2', 'val-ood-c3', 'val-ood-c4'
                        #    ],
                        batch_size = bs,
                        n_threads = n_threads,
                        verbose=verbose
                        )
