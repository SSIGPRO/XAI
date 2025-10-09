import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

# Python stuff
from functools import partial
from time import time
import argparse
import torch
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd_fns import linear_svd
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
parser.add_argument("--ci", required=True, type=str, help="Corruption intensity")
args = parser.parse_args()

emb_size = args.emb_size
ci = args.ci

if ci == 'high': 
    from config_anomalies import ch as corruptions
elif ci == 'medium':
    from config_anomalies import cm as corruptions
elif ci == 'low':
    from config_anomalies import cl as corruptions
else:
    raise RuntimeError('The configuration is not available choose among [low|medium|high]')

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
    model_name = "conv2dAE_SENT_L16_K3-3_Emblarge_Lay0_C16_S42.pth"

    parsed_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_prova')

    svds_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/') 
    svds_name = 'svds' 
    
    cvs_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/corevectors_prova')
    cvs_name = 'cvs'

    loaders = [
            'val',

            # f'val-val-c-single-{ci}',
            # f'val-test-c-single-{ci}',
            f'test-val-c-single-{ci}',
            f'test-test-c-single-{ci}',

            # f'val-val-c-all-{ci}',
            # f'val-test-c-all-{ci}',
            f'test-val-c-all-{ci}',
            f'test-test-c-all-{ci}',

            # f'val-val-c-RW-{ci}',
            # f'val-test-c-RW-{ci}',
            f'test-val-c-RW-{ci}',
            f'test-test-c-RW-{ci}',
            ]

    bs = 2**18
    verbose = True 
    n_threads = 1
    input_key = 'data'

    n_classes = len(corruptions.keys())
    
    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False   

    target_layer = 'encoder.linear'
    layer_svd_rank = 300

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
            target_modules = [target_layer]
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
        
        svd_fns = {
            target_layer: partial(
            linear_svd, 
            rank = layer_svd_rank,
            device = device,
            ),
        }

        t0 = time()
        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = [target_layer],
                sample_in = sentinel._dss[loaders[0]][0][input_key],
                svd_fns = svd_fns,
                verbose = verbose
                )
        print('time: ', time()-t0)

    #--------------------------------
    # Corevectors                              
    #--------------------------------
    
    # get the core vectors
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
    )

    reduction_fns = {
            target_layer: partial(
                linear_svd_projection, 
                svd = model._svds[target_layer], 
                layer = model._target_modules[target_layer], 
                use_s = True,
                device=device
                )
    }

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
                  
