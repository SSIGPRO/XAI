import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff 

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
parser.add_argument("--n_cluster", required=True, help="number of cluster for the cvs")
args = parser.parse_args()

n_cluster = args.n_cluster

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
    bs = 2**12 
    n_threads = 1

    cvs_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name = 'corevectors'

    embeds_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    embeds_name = 'CLIP_embeddings_ViT-L14'

    drill_path = Path.cwd()/f'../../data/{model_name}/drillers'
    drill_name = 'classifier'
    
    verbose = True 
    
    '''
    print('\n----------- svds:')
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
        s = model._svds[k]['s']
        if len(s.shape) == 1:
            plt.figure()
            plt.plot(s, '-')
            plt.xlabel('Rank')
            plt.ylabel('EigenVec')
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
            for r in range(s.shape[0]):
                plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Channel')
            ax.set_zlabel('EigenVec')
        plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
    '''

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    random_subsampling(ds, 0.3)
    
    # corevecs = CoreVectors(
    #         path = cvs_path,
    #         name = cvs_name,
    #         model = model,
    #         )
    
    embeds = CoreVectors(
            path = embeds_path,
            name = embeds_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    
    with embeds as em: #corevecs as cv,

        # cv.parse_ds(
        #         batch_size = bs,
        #         datasets = ds,
        #         n_threads = n_threads,
        #         verbose = verbose
        #         )
        
        em.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )

        '''
        # This occupies a lot of space. Only do if you need it
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )        
        ''' 

        em.get_clip_embeddings(device=device, 
                               clip_model='ViT-L/14', 
                               batch_size=bs, 
                               n_threads=n_threads, 
                               verbose=verbose)
        quit()
        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )

        #if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
        cv.normalize_corevectors(
                wrt = 'train',
                #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                to_file = cvs_path/(cvs_name+'.normalization.pt'),
                batch_size = bs,
                n_threads = n_threads,
                verbose=verbose
                )
        
        fig, axs = plt.subplots(1,2, figsize=(12, 6))
        axs[0].hist(cv._dss['train']['label'], bins=1000, color='blue', alpha=0.7)
        axs[0].set_title('Train Labels Distribution')
        axs[0].set_xlabel('Labels')
        axs[0].set_ylabel('Frequency')
        axs[1].hist(cv._dss['train']['label'][cv._dss['train']['result']==0], bins=1000, color='red', label='W',alpha=0.7)
        axs[1].hist(cv._dss['train']['label'][cv._dss['train']['result']==1], bins=1000, color='green', label='C',alpha=0.7)
        axs[1].set_title('Correct and Wrong Labels Distribution')
        axs[1].set_xlabel('Labels')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        plt.tight_layout()
        plt.savefig((cvs_path/(cvs_name+'.labels_distribution.png')).as_posix(), dpi=300, bbox_inches='tight')
        plt.close() 
       
    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    n_cluster = 1000

    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parsers[peep_layer],
                device = device
                )

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'val'],
                verbose = True
                )
        for ds_key, cv_dss in cv._dss.items():
            print(ds_key, cv_dss['result'].sum()/len(cv_dss))

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(corevectors = cv, verbose=verbose)
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        corevectors = cv,
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()
    
