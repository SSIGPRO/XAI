import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

# Python stuff
from functools import partial
import argparse
import torch
from cuda_selector import auto_cuda
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

from config_cv_ph import *

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
args = parser.parse_args()

emb_size = args.emb_size

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
    model_name = f"conv2dAE_SENT_L16_K3-3_Emb{emb_size}_Lay0_C16_S42.pth"

    parsed_path = Path(f'/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_{emb_size}_all')

    svds_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/') 
    svds_name = f'svds_{emb_size}' 
    
    cvs_path = Path(f'/srv/newpenny/XAI/generated_data/AE_sentinel/corevectors_{emb_size}_all')
    cvs_name = 'cvs'

    loaders = [
            'val', 'test_ori',

            # f'val-val-c-single-high',
            # f'val-test-c-single-high',
            # f'test-val-c-single-high',
            # f'test-test-c-single-high',

            f'val-val-c-all-high',
            f'val-test-c-all-high',
            f'test-val-c-all-high',
            f'test-test-c-all-high',

            # f'val-val-c-RW-high',
            # f'val-test-c-RW-high',
            # f'test-val-c-RW-high',
            # f'test-test-c-RW-high',

            # f'val-val-c-single-medium',
            # f'val-test-c-single-medium',
            # f'test-val-c-single-medium',
            # f'test-test-c-single-medium',

            # f'val-val-c-all-medium',
            # f'val-test-c-all-medium',
            # f'test-val-c-all-medium',
            # f'test-test-c-all-medium',

            # f'val-val-c-RW-medium',
            # f'val-test-c-RW-medium',
            # f'test-val-c-RW-medium',
            # f'test-test-c-RW-medium',

            # f'val-val-c-single-low',
            # f'val-test-c-single-low',
            # f'test-val-c-single-low',
            # f'test-test-c-single-low',

            # f'val-val-c-all-low',
            # f'val-test-c-all-low',
            # f'test-val-c-all-low',
            # f'test-test-c-all-low',

            # f'val-val-c-RW-low',
            # f'val-test-c-RW-low',
            # f'test-val-c-RW-low',
            # f'test-test-c-RW-low',
            ]

    layer_svd_rank = 300
    input_key = 'data'

    #--------------------------------
    # Model
    #--------------------------------
 
    sentinel_model = CONV_AE2D(
              num_sensors=num_sensors,
              seq_len=seq_len,
              kernel_size=kernel,
              embedding_size=emb_size,
              lay3=lay3
          )
    print(sentinel_model)
    quit()
        
    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    model.load_checkpoint(
            path = model_path,
            name = model_name,
            sd_key = 'model_state_dict'
            )

    model.set_target_modules(
            target_modules = target_layers
            )

    #--------------------------------
    # Dataset                              
    #--------------------------------

    sentinel = Sentinel(
            path = parsed_path
            )

    #--------------------------------
    # SVDs                              
    #--------------------------------
    with sentinel as s:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )
        
        svd_fns = {}
        for _layer in linear_layers: 
            svd_fns[_layer] = partial(
                linear_svd, 
                rank = layer_svd_rank,
                device = device,
                )
        for _layer in conv_layers:   
            svd_fns[_layer] = partial(
                conv2d_toeplitz_svd, 
                rank = layer_svd_rank,
                channel_wise = False,
                device = device,
                )

        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = sentinel._dss[loaders[0]][0][input_key],
                svd_fns = svd_fns,
                verbose = verbose
                )
        print('time: ', time()-t0)
        plt.plot(model._svds[target_layer]['s'])
        plt.savefig(Path.cwd()/f'temp_plots_{emb_size}/svd_profile.png')

    #--------------------------------
    # Corevectors                              
    #--------------------------------
    
    # get the core vectors
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
    )

    reduction_fns = {}
    for _layer in linear_layers:
        reduction_fns[_layer] = partial(
                linear_svd_projection, 
                svd = model._svds[_layer], 
                layer = model._target_modules[_layer], 
                use_s = True,
                device=device
                )
    for _layer in conv_layers:
        reduction_fns[_layer] = partial(
                conv2d_toeplitz_svd_projection, 
                svd = model._svds[_layer], 
                layer = model._target_modules[_layer], 
                use_s = True,
                device=device
                )

    with sentinel as s, corevecs as cv:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )
        
        cv.get_coreVectors(
            datasets = s,
            loaders = loaders,
            input_key = input_key,
            reduction_fns = reduction_fns,
            batch_size = bs,
            verbose = verbose
        )
         
        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'val',
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    batch_size = bs,
                    verbose=verbose
                    )
                  