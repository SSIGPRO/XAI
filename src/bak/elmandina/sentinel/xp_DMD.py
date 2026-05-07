from functools import partial
import sys
from time import time
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance

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



from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD 

from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv as CWM

from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection
from peepholelib.models.svd_fns import conv2d_toeplitz_svd, linear_svd, conv2d_kernel_svd
#from peepholelib.scores.mahalanobis import mahalanobis_score

from xp_corruption import corruptions
from xp_sentinel import sentinel_model
from xp_scores import *
from load_only import feature_sizes, target_layers

#import xp_corruption
#print(dir(xp_corruption))
#quit()
def get_output(**kwargs):
    return kwargs['dss']['output']

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #print(sentinel_model)
    #quit()
    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = Path.cwd()/'../../data/datasets'

    cvs_path = Path.cwd()/'../../data/corevectors'
    cvs_name = 'cvs'

    dmd_drill_path = Path.cwd()/'../../data/drillers'
    dmd_drill_name = 'drill_dmd'

    dmd_phs_path = Path.cwd()/'../../data/peepholes'
    dmd_phs_name = 'peepholes_dmd'

    magnitude = 0
    loaders = ['train', 'val', 'test', 'val-c-all-GWM']
    bs = 2**20 #1024*5*10
    verbose = True 
    input_key = 'data'
    n_threads = 1
    n_cluster = 50
    n_classes = len(corruptions) #this is the number of clusters is the number of corruptions 

    target_layers = ['encoder.linear', 'decoder.nn_dec_body.linear', 'encoder.nn_enc_body.layer1.conv1']

    ###############################
    #   Model
    ###############################
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

    ###############################
    #   Get Corevectors
    ###############################
    corevectors = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        loaders = loaders,#['val-c-all'],#-c-all-channels
        verbose = verbose
    )

    for test_name, test_key in tests.items():
        #print(f'test_name{test_name}\test_key{test_key}')
        #quit()
        if tests[test_name]['empp_fit_key'] != None:
            drillers = {}
            for layer in target_layers:
                drillers[layer] = DMD(
                    path = dmd_drill_path,
                    name = dmd_drill_name+f'.{layer}.{test_name}',
                    nl_model = test_key['n_classes'],
                    n_features = feature_sizes[layer],
                    parser = get_output, #s._dss['output'],
                    model= model,
                    layer = layer,
                    magnitude = magnitude,
                    std_transform = [0.300, 0.287, 0.294],
                    parser_act = CWM,
                    device = device
                )
            peepholes_dmd = Peepholes(
                path = dmd_phs_path,
                name = dmd_phs_name+f'.{test_name}.{test_key['label_key']}',
                device = device
            )

            # get peeps
            with sentinel as s, corevectors as cv:
                s.load_only(
                    loaders = loaders,#test_key['loaders'],
                    verbose = verbose
                )
                
                cv.load_only(
                    loaders = loaders, #test_key['loaders'],
                    verbose = verbose
                )
                #print(type(drillers))
                #quit()
                for layer, driller in drillers.items():
                    
                    if(dmd_drill_path/driller._suffix/'precision.pt').exists():
                        driller.load()
                    else:
                        driller.fit(
                            dataset = s,
                            corevectors = cv,
                            loader = loaders, #test_key['empp_fit_key'],
                            drill_key = layer,
                            label_key = test_key['label_key'],
                            verbose = verbose
                        )
                        driller.save()
                
                with peepholes_dmd as ph:
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
                        
                            peeps = ph._phs[k][kk]#['encoder.linear']
                            print(f'peepsDMD{peeps}')
    
