from functools import partial
import sys
from time import time
from matplotlib import pyplot as plt
import numpy as np

import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())
#from sentinel.xp_corruption import corruptions

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import conv2d_toeplitz_svd, linear_svd


from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection

from xp_corruption import corruptions
#import xp_corruption
#print(dir(xp_corruption))
#quit()

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = Path.cwd()/'../../data/datasets'
    svds_path = Path.cwd()/'../../data'
    svds_name = 'svds' 

    pca_path = Path.cwd()/'../../data/datasets'
    pca_name = 'pcas'
    
    cvs_path = Path.cwd()/'../../data/corevectors'
    cvs_name = 'cvs'

    
    phs_path = Path.cwd()/'../../data/peepholes'
    phs_name = 'peepholes'

    drill_path = Path.cwd()/'../../data/drillers'
    drill_name = 'drills'

    loaders = ['train', 'val', 'test']
    bs = 2**20 #1024*5*10
    verbose = True 
    input_key = 'data'
    n_threads = 8
    n_cluster = 50
    n_classes = len(corruptions) #this is the number of clusters is the number of corruptions 
    #model = "ae1Dregn16_2_ns0_k001.pth"

    model_dir = '/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth'
    
    #--------------------------------
    # Model
    #--------------------------------
    
    sentinel_model = torch.load("/srv/newpenny/XAI/EP/Model_ready/ae1Dregn16_2_ns0_k001.pth",map_location=device, weights_only=False)
    target_layers = ['encoder.linear']

    layer_svd_rank = 100
    layer_cv_dim = 5
       
    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    sentinel = Sentinel(
        path = parsed_path
    )

    model.set_target_modules(
        target_modules = target_layers
    )
    
    with sentinel as s:
        s.load_only(
            loaders = ['val'],#val-c /-c-all-channels
            verbose = verbose
        )
        #print(s._dss['val']['latent_space'].shape)
        #quit()
        svd_fns = {
            'encoder.linear': partial(
            linear_svd, 
            rank = layer_svd_rank,
            device = device,
            ),
        }

        t0 = time()
        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = sentinel._dss['val'][0]['data'],#-c-all-channels
                svd_fns = svd_fns,
                verbose = verbose
                )
        print('time: ', time()-t0)

        for k in model._svds.keys():
            for kk in model._svds[k].keys():
                #print(f'model._svds[k].keys()={kk}')
                print('svd shapes: ', k, kk, model._svds[k][kk].shape)
            s = model._svds[k]['s']
            print(f's={s}')

            fig = plt.figure()
            plt.plot(s)
            plt.savefig('svd_lin_lay_profile.png')
            plt.close()
    
    with sentinel as s:
        s.load_only(
            loaders = ['val'],#val-c /-c-all-channels
            verbose = verbose
        )
        model.get_pcas(
            path = pca_path,
            name = pca_name,
            target_modules = target_layers,
            sample_in = s._dss['val']['latent_space'],
            verbose= verbose
        )
        for k in model._pcas.keys():   # k = 'latent_space'
            for kk in model._pcas[k].keys():  # kk = 'Vt', 'S'
                print(f'Pca shapes: {k}, {kk}, {model._pcas[k][kk].shape}')

            s_ = model._pcas[k]['S']
            print(f's_:{s_}')

            fig = plt.figure()
            plt.plot(s_.cpu().numpy())
            plt.savefig('pca_latent_profile.png')
            plt.close()
        quit()


    # get the core vectors
    cvs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        loaders = ['val'],#-c-all-channels
        verbose = verbose
    )

    reduction_fns = {
            'encoder.linear': partial(
                linear_svd_projection, 
                svd = model._svds['encoder.linear'], 
                layer = model._target_modules['encoder.linear'], 
                use_s = True,
                device=device
                ),
    }

    with sentinel as s, cvs as cv:
        s.load_only(
            loaders = ['val'],#loaders,-c-all-channels
            verbose = verbose
        )
        
        cv.get_coreVectors(
            datasets = s,#sentinel,
            loaders = ['val'],#loaders,-c-all-channels
            input_key = input_key,
            reduction_fns = reduction_fns,
            batch_size = bs,
            verbose = verbose
        )
        
        
        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'val',#sentinel._dss['train'][0:1][input_key], val-c only to test: was val
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #loaders = ['CIFAR100-val', 'CIFAR100-test'],
                    batch_size = bs,
                    #n_threads = n_threads,
                    verbose=verbose
                    )
        
        for k in cv._corevds.keys():
            for kk in cv._corevds[k].keys():
                print('cv shapes: ', k, kk, cv._corevds[k][kk].shape)
            
            v = cv._corevds[k]#['encoder.linear']
            
            #plotting the cv just to see how looks like
            # plot core vector 1
            #plt.plot(v['encoder.linear'][0])
            #plt.savefig('linear_l_cv_trial2.png')
            #plt.close()
        
            '''
            fig = plt.figure()
            plt.plot(v['encoder.linear'].cpu())
            plt.savefig('linear_l_cv.png')
            plt.close()
            '''
    quit()
    # drillers 
    #TODO see how GMM works from the name seems gaussian mixture  
    cv_parsers = {
        'encoder.linear': partial(
            trim_corevectors,
            module = 'encoder.linear',
            cv_dim = layer_cv_dim,
            label_key = 'corruption'
        )
    }

    feature_sizes ={
        'encoder.linear':layer_cv_dim
    }

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
    
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )
    
    # fitting the classifiers
    with sentinel as s, cvs as cv:
        s.load_only(
            loaders = ['val-c-all-channels'],
            verbose = verbose
        )

        cv.load_only(
            loaders = ['val-c-all-channels'],
            verbose = verbose
        )

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                v = driller.load()
            
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(
                        corevectors = cv,
                        loader = 'val-c-all-channels',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)
                print(f'driller={drill_key}{driller}')

                driller.compute_empirical_posteriors(
                        datasets = s,
                        corevectors = cv,
                        loader = 'val-c-all-channels',
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()
            
            #since _emmp is tensor to visualize needed to convert to numpy 1st
            #TODO find a way to plot on vscode and not save as img
            d = driller._empp.detach().cpu().numpy()
            plt.imsave('empp.png',d)

    # compute the peepholes
    with sentinel as s, cvs as cv, peepholes as ph:
        s.load_only(
                loaders = ['val-c-all-channels'],
                verbose = verbose
                )
        print(f'See the corruptions:{s._dss['val-c-all-channels']['corruption']}')
        quit()

        cv.load_only(
                loaders = ['val-c-all-channels'],
                verbose = verbose 
                ) 

        ph.get_peepholes(
                datasets = s,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
            )
        
        

        for k in ph._phs.keys():
            for kk in ph._phs[k].keys():
                print('ph shapes: ', k, kk, ph._phs[k][kk].shape)
            
            peeps = ph._phs[k]#['encoder.linear']
            '''
            fig = plt.figure()
            plt.plot(peeps['encoder.linear']['peepholes'].cpu())
            plt.savefig('peephole_trial.png')
            plt.close()
            '''