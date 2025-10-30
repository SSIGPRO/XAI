import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap, means, stds 
from peepholelib.coreVectors.coreVectors import CoreVectors as Embeddings

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
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    with dataset as ds: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )
        
        #--------------------------------
        # CLIP Embeddings 
        #--------------------------------
        
        embeds = Embeddings(
                path = emb_path,
                name = emb_name,
                model = model,
                )
        
        with embeds as em: 
                    
            # computing the corevectors
            em.get_clip_embeddings(
                    datasets = ds,
                    device=device, 
                    clip_model='ViT-L/14', 
                    batch_size=bs, 
                    n_threads=n_threads, 
                    verbose=verbose
                    )