import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.utils.fine_tune import fine_tune 

# python stuff
import argparse

# torch stuff
import torch
from cuda_selector import auto_cuda

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
args = parser.parse_args()
emb_size = args.emb_size

def get_input(x):
    ret = {
            'image': x['data'],
            'label': x['data']
            }
    return ret

def get_output(x):
    return x[0]

def acc_fn(pred, targets):
    return (pred-targets).norm()

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
    model_name = "conv2dAE_SENT_L16_K3-3_Emblarge_Lay0_C16_S42.pth"
    parsed_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/datasets')

    # model parameters
    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False   
    
    #tune_dir = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/train_cps')
    tune_dir = Path.cwd()/'../../data/train_cps'
    tune_name = 'checkpoints'

    verbose = True 
    # TODO: check
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

    model = ModelWrap(
            model = sentinel_model,
            device = device
            )

    #--------------------------------
    # Dataset 
    #--------------------------------
    
    # Assuming we have a parsed dataset in ds_path
    datasets = Sentinel(
            path = parsed_path,
            )

    #--------------------------------
    # Tuning 
    #--------------------------------
    with datasets as ds:    
        ds.load_only(
                loaders = ['train', 'val'],
                verbose = verbose
                )

        fine_tune(
                path = tune_dir,
                name = tune_name,
                model = model,
                dataset = ds,
                train_key = 'train',
                val_key = 'val',
                in_parser = get_input,
                out_parser = get_output,
                loss_fn = torch.nn.MSELoss,
                loss_kwargs = {'reduction': 'mean'},
                optimizer = torch.optim.Adam,
                optim_kwargs = {},
                acc_fn = acc_fn,
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
                sched_kwargs = dict(
                    mode = "min",
                    factor = 0.8,
                    patience = 5,
                    threshold = 1e-4,
                    threshold_mode = "rel",
                    cooldown = 0,
                    min_lr = 9e-8
                    ),
                lr = 3e-3,
                iterations = 'full',
                batch_size = 2**12,
                max_epochs = 200,
                save_every = 50,
                n_threads = 1,
                verbose = verbose
                )
