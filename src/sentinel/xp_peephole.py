import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())

# Python stuff
from functools import partial
from time import time
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from cuda_selector import auto_cuda

# Our stuff
from peepholelib.datasets.sentinel import Sentinel
from peepholelib.models.model_wrap import ModelWrap

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.parsers import trim_corevectors

from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM
from peepholelib.peepholes.peepholes import Peepholes 

from peepholelib.plots.conceptograms import plot_conceptogram
from peepholelib.utils.topk_acc import compute_top_k_accuracy

from peepholelib.models.sentinel.model_sentinel import CONV_AE2D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
parser.add_argument("--ci", required=True, type=str, help="Corruption intensity")
parser.add_argument("--fit", required=True, type=str, help="Corruption intensity")
args = parser.parse_args()

emb_size = args.emb_size
ci = args.ci
fit = args.fit

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
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
    model_name = "conv2dAE_SENT_L16_K3-3_Emblarge_Lay0_C16_S42.pth"

    parsed_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/datasets')

    svds_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/') 
    svds_name = 'svds' 
    
    cvs_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/corevectors')
    cvs_name = 'cvs'

    drill_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/drillers')
    drill_name = 'classifier'

    phs_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/peepholes')
    phs_name = 'peepholes'

    plots_path = Path.cwd()/f'temp_plots'
    plots_path.mkdir(parents=True, exist_ok=True)

    bs = 2**18
    verbose = True 
    n_threads = 1

    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False  
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    target_layers = ['encoder.linear']
    
    cv_dims = [10, 100, 200, 50]#, 2, 
    n_clusters = [5, 10, 15, 20, 50] #

    tests = {
            'single_channel': {
                'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
                'empp_fit_key': f'{fit}-val-c-single-{ci}', 
                'label_key': 'channel',
                'n_classes': 16,
                'class_names': [f'Ch{i}' for i in range(16)] 
                },
            'single_corruption': {
                'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
                'empp_fit_key': f'{fit}-val-c-single-{ci}', 
                'label_key': 'corruption',
                'n_classes': len(corruptions.keys()),
                'class_names': corruptions.keys()
                },
            'single_RW': {
                'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
                'empp_fit_key': f'{fit}-val-c-single-{ci}', 
                'label_key': 'RW',
                'n_classes': 4, 
                'class_names': [f'RW{i}' for i in range(4)] 
                },
            'all': {
                'loaders': [f'{fit}-val-c-all-{ci}', f'{fit}-test-c-all-{ci}'],
                'empp_fit_key': f'{fit}-val-c-all-{ci}', 
                'label_key': 'corruption',
                'n_classes': len(corruptions.keys()),
                'class_names': corruptions.keys()
                },
            'RW_corruption': {
                'loaders': [f'{fit}-val-c-RW-{ci}', f'{fit}-test-c-RW-{ci}'],
                'empp_fit_key': f'{fit}-val-c-RW-{ci}', 
                'label_key': 'corruption',
                'n_classes': len(corruptions.keys()),
                'class_names': corruptions.keys() 
                },
            'RW_RW': {
                'loaders': [f'{fit}-val-c-RW-{ci}', f'{fit}-test-c-RW-{ci}'],
                'empp_fit_key': f'{fit}-val-c-RW-{ci}', 
                'label_key': 'RW',
                'n_classes': 4,
                'class_names': [f'RW{i}' for i in range(4)] 
                },
            }

    #--------------------------------
    # Dataset                              
    #--------------------------------

    sentinel = Sentinel(
            path = parsed_path
            )

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
    )

    #--------------------------------
    # Peepholes
    #--------------------------------
    for cv_dim in cv_dims:
        for n_cluster in n_clusters:
            for test_name in tests:
                # parsers and shit
                cv_parsers = {
                        _layer: partial(
                            trim_corevectors,
                            module = _layer,
                            cv_dim = cv_dim,
                            label_key = tests[test_name]['label_key'] 
                            ) for _layer in target_layers
                        }

                feature_sizes = {_layer: cv_dim for _layer in target_layers}

                drillers = {}
                for peep_layer in target_layers:
                    drillers[peep_layer] = tGMM(
                            path = drill_path,
                            name = drill_name+f'.{fit}.{peep_layer}.{test_name}.{emb_size}.{ci}',
                            nl_classifier = n_cluster,
                            nl_model = tests[test_name]['n_classes'],
                            n_features = feature_sizes[peep_layer],
                            parser = cv_parsers[peep_layer],
                            device = device
                            )

                peepholes = Peepholes(
                        path = phs_path,
                        name = phs_name+f'.{fit}.{n_cluster}.{cv_dim}.{test_name}.{emb_size}.{ci}',
                        device = device
                        )

                # get peepholes
                with sentinel as s, corevecs as cv:
                    s.load_only(
                            loaders = tests[test_name]['loaders'],
                            verbose = verbose
                            )

                    cv.load_only(
                            loaders = tests[test_name]['loaders'],
                            verbose = verbose 
                            ) 

                    for drill_key, driller in drillers.items():
                        if (driller._empp_file).exists():
                            print(f'Loading Classifier for {drill_key}') 
                            driller.load()
                            plt.imshow(driller._empp.cpu())
                            plt.savefig(driller._clas_path / f'{drill_key}_empp.png')
                        else:
                            t0 = time()
                            print(f'Fitting classifier for {drill_key}')
                            driller.fit(
                                    corevectors = cv,
                                    loader = tests[test_name]['empp_fit_key'],
                                    verbose=verbose
                                    )
                            print(f'Fitting time for {drill_key}  = ', time()-t0)

                            driller.compute_empirical_posteriors(
                                    datasets = s,
                                    corevectors = cv,
                                    loader = tests[test_name]['empp_fit_key'],
                                    batch_size = bs,
                                    verbose=verbose
                                    )
                    
                            # save classifiers
                            print(f'Saving classifier for {drill_key}')
                            driller.save()
                            plt.imshow(driller._empp.cpu())
                            plt.savefig(driller._clas_path / f'{drill_key}_empp.png')
                
                        with peepholes as ph:
                            s.load_only(
                                    loaders = tests[test_name]['loaders'],
                                    verbose = verbose
                                    )

                            cv.load_only(
                                    loaders = tests[test_name]['loaders'],
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
                            
                            for _layer in target_layers:
                                cns = tests[test_name]['class_names']
                                fig, axs = plt.subplots(1, 2, figsize=(20, 8))

                                for loader_n, loader_key in enumerate(tests[test_name]['loaders']):
                                    # the data 
                                    idx = s._dss[loader_key]['detection'] == 1
                                    result = ph._phs[loader_key][_layer]['peepholes'][idx]
                                    label = s._dss[loader_key][tests[test_name]['label_key']][idx]
                                    
                                    # confusion matrix
                                    cm = confusion_matrix(label, result.argmax(dim=1), normalize='true')
                                    disp = ConfusionMatrixDisplay(cm, display_labels=cns)
                                    disp.plot(ax=axs[loader_n], cmap='Blues', colorbar=False, values_format=".2f", im_kw={'norm': norm})

                                    # text around
                                    axs[loader_n].set_title(f"{loader_key.capitalize()} set")
                                    axs[loader_n].tick_params(axis='x', rotation=45)
                                fig.suptitle(f'Confusion Matrix {_layer}: cv_dim={cv_dim} & n_cluster={n_cluster}')
                                plt.tight_layout()
                                plt.savefig(Path(plots_path)/f"CM.{fit}.{test_name}.{n_cluster}.{cv_dim}.{emb_size}.{ci}.png", bbox_inches='tight', dpi=300)
                                plt.close()
