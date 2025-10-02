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

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.datasets.sentinel import SentinelWrap
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import conv2d_toeplitz_svd, linear_svd

from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection


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
    
    cvs_path = Path.cwd()/'../../data/corevectors'
    cvs_name = 'cvs'

    drill_path = Path.cwd()/'../../data/drillers'
    drill_name = 'drills'

    loaders = ['train', 'val', 'test']
    bs = 1024*5*10
    verbose = True 
    input_key = 'data'
    n_cluster = 5
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
            loaders = ['val', 'val-c'],
            verbose = verbose
        )
        
        svd_fns = {
            'encoder.linear': partial(
            linear_svd, 
            rank = layer_svd_rank,
            device = device,
            ),
        }

        #print(f'checking what sentinel._dss[train][data] is: {sentinel._dss['train']['data']}')
        #quit()

        t0 = time()
        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = sentinel._dss['train'][0]['data'],
                svd_fns = svd_fns,
                verbose = verbose
                )
        print('time: ', time()-t0)

        for k in model._svds.keys():
            #print(f'model._svds.keys()={k}')
            for kk in model._svds[k].keys():
                #print(f'model._svds[k].keys()={kk}')
                print('svd shapes: ', k, kk, model._svds[k][kk].shape)
            s = model._svds[k]['s']
            print(f's={s}')

            #fig = plt.figure()
            #plt.plot(s)
            #plt.savefig('aaaa.png')
            #plt.close()



    # get the core vectors
    cvs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        loaders = loaders,
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
            loaders = loaders,
            verbose = verbose
        )
        
        cv.get_coreVectors(
            datasets = s,#sentinel,
            loaders = loaders,
            input_key = input_key,
            reduction_fns = reduction_fns,
            batch_size = bs,
            verbose = verbose
        )
        
        
        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'train',#sentinel._dss['train'][0:1][input_key],
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
            print(f'v={v}')
            fig = plt.figure()
            # plot core vector 1
            plt.plot(v['encoder.linear'][0])
            plt.savefig('linear_l_cv_trial2.png')
            plt.close()
        
            '''
            fig = plt.figure()
            plt.plot(v['encoder.linear'].cpu())
            plt.savefig('linear_l_cv.png')
            plt.close()
            '''

    # drillers 
    #TODO see how GMM works from the name seems gaussian mixture  
    #TODO drillers peepholesthis will be done after implementation of labels

    cv_parsers = {
        'encoder.linear': partial(
            trim_corevectors,
            module = 'encoder.linear',
            cv_dim = layer_cv_dim
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
 