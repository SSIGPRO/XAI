import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import clip

# Our stuff 

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
args = parser.parse_args()

if sys.argv[1] == 'vgg':
    from config.config_vgg import *
elif sys.argv[1] == 'vit':
    from config.config_vit import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg_cifar10|vgg_cifar100|vit_cifar100>\'')


if __name__ == "__main__":
    
    #--------------------------------
    # Directories definitions
    #--------------------------------

    # model parameters 
    seed = 29
    bs = 512 
    n_threads = 1

    model_name = 'vgg'

    cvs_path_vgg = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name_vgg = 'corevectors'

    embeds_path_vgg = Path.cwd()/f'../../data/{model_name}/corevectors'
    embeds_name_vgg = 'CLIP_embeddings_ViT-L14'

    model_name = 'vit'

    cvs_path_vit = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name_vit = 'corevectors'

    embeds_path_vit = Path.cwd()/f'../../data/{model_name}/corevectors'
    embeds_name_vit = 'CLIP_embeddings_ViT-L14'
    
    verbose = True 

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    model, preprocess = clip.load("ViT-L/14", device=device)

    random_subsampling(ds, 0.3)
    
    corevecs_vgg = CoreVectors(
            path = cvs_path_vgg,
            name = cvs_name_vgg,
            model = model,
            )
    
    corevecs_vit = CoreVectors(
            path = cvs_path_vit,
            name = cvs_name_vit,
            model = model,
            )
    
    embeds_vit = CoreVectors(
            path = embeds_path_vit,
            name = embeds_name_vit,
            model = model,
            )
    
    embeds_vgg = CoreVectors(
            path = embeds_path_vgg,
            name = embeds_name_vgg,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    
    with corevecs_vgg as cvg, corevecs_vit as cvt, embeds_vgg as emg, embeds_vit as emt: 

        cvg.load_only(
            loaders = ['train', 'val'],
            verbose = verbose
        )

        cvt.load_only(
            loaders = ['train', 'val'],
            verbose = verbose
        )

        emg.load_only(
            loaders = ['train', 'val'],
            verbose = verbose
        )

        emt.load_only(
            loaders = ['train', 'val'],
            verbose = verbose
        )

        imgg = cvg._dss['train']['image'][0]
        
        imgt = cvt._dss['train']['image'][0]
        
        fig, axs = plt.subplots(1,2)
        axs[0].imshow(imgg.permute(1,2,0))
        axs[0].set_title('VGG-16')
        axs[1].imshow(imgt.permute(1,2,0))  
        axs[1].set_title('ViT-L/14')
        plt.savefig('example_images.png')
        plt.show()