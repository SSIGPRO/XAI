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
from tqdm import tqdm

# Torch stuff
import torch
from torch.utils.data import DataLoader
from cuda_selector import auto_cuda
import argparse

# Our stuff
from peepholelib.models.sentinel.model_sentinel import CONV_AE2D
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.sentinel import Sentinel

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.parsers import trim_corevectors

from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM
from peepholelib.peepholes.peepholes import Peepholes 

from peepholelib.utils.topk_acc import compute_top_k_accuracy

from config_cv_ph import *
#from classify_cps import classify_cps #  moved to bak

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
parser.add_argument("--ci", required=True, type=str, help="Corruption intensity")
parser.add_argument("--fit", required=True, type=str, help="'val' or 'test'")
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

def emp_single_label(**kwargs):
    '''
    New function for the implementation of new empirical posterior
    '''

    dss = kwargs.get('datasets')
    cvs = kwargs.get('corevectors')
    loader = kwargs.get('loader', 'train')
    bs = kwargs.get('batch_size', 64)
    verbose = kwargs.get('verbose', False)
    nl_class = kwargs('n_cluster')
    nl_model = kwargs('n_clasess')
    parser = kwargs('parser')
    driller = kwargs('driller')
    device = kwargs('device')

    _empp = torch.zeros(nl_model, 2, nl_class)

    dss_dl = DataLoader(dss._dss[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)
    cvs_dl = DataLoader(cvs._corevds[loader], batch_size=bs, collate_fn=lambda x: x, shuffle=False)

    if verbose: print('Computing empirical posterior')
    for _dss, _cvs in tqdm(zip(dss_dl, cvs_dl), disable=not verbose):
        data, label = parser(cvs=_cvs, dss=_dss)
        data, label = data.to(device), label.to(device)
        label_one_hot = torch.functional.one_hot(label, num_classes=nl_model)
        print(label_one_hot)
        preds = driller.predict(data)

        for loh, p in zip(label_one_hot, preds):
            for i, c in enumerate(loh):
                _empp[i, c, p] += 1

        _empp = _empp[:, 1, :].reshape()


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
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

    drill_path = Path(f'/srv/newpenny/XAI/generated_data/AE_sentinel/drillers_{emb_size}_all')
    drill_name = 'classifier'

    phs_path = Path(f'/srv/newpenny/XAI/generated_data/AE_sentinel/peepholes_{emb_size}_all')
    phs_name = 'peepholes'

    plots_path = Path.cwd()/f'temp_plots_{emb_size}'
    plots_path.mkdir(parents=True, exist_ok=True)

    bs = 2**20
    verbose = True 
    n_threads = 1

    num_sensors = 16
    seq_len = 16
    kernel = [3, 3]
    lay3 = False  
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 
    
    cv_dims = [50]#, 2, 10, 100, 200, 
    n_clusters = [50] #5, 10, 15, 20,

    tests = {
            # 'single_channel': {
            #     'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
            #     'empp_fit_key': f'{fit}-val-c-single-{ci}', 
            #     'label_key': 'channel',
            #     'n_classes': 16,
            #     'class_names': [f'Ch{i}' for i in range(16)] 
            #     },
            # 'single_corruption': {
            #     'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
            #     'empp_fit_key': f'{fit}-val-c-single-{ci}', 
            #     'label_key': 'corruption',
            #     'n_classes': len(corruptions.keys()),
            #     'class_names': corruptions.keys()
            #     },
            # 'single_RW': {
            #     'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
            #     'empp_fit_key': f'{fit}-val-c-single-{ci}', 
            #     'label_key': 'RW',
            #     'n_classes': 4, 
            #     'class_names': [f'RW{i}' for i in range(4)] 
            #     },
            'all': {
                'loaders': [f'{fit}-val-c-all-{ci}', f'{fit}-test-c-all-{ci}', 'test_ori'],
                'empp_fit_key': f'{fit}-val-c-all-{ci}', 
                'label_key': [f'corruption{i}' for i in range(len(corruptions.keys()))],
                'n_classes': 2,#len(corruptions.keys()),
                'class_names': corruptions.keys()
                },
            # 'RW_corruption': {
            #     'loaders': [f'{fit}-val-c-RW-{ci}', f'{fit}-test-c-RW-{ci}', 'test_ori'],
            #     'empp_fit_key': f'{fit}-val-c-RW-{ci}', 
            #     'label_key': 'corruption',
            #     'n_classes': len(corruptions.keys()),
            #     'class_names': corruptions.keys() 
            #     },
            # 'RW_RW': {
            #     'loaders': [f'{fit}-val-c-RW-{ci}', f'{fit}-test-c-RW-{ci}', 'test_ori'],
            #     'empp_fit_key': f'{fit}-val-c-RW-{ci}', 
            #     'label_key': ['RW0','RW1', 'RW2', 'RW3'],
            #     'n_classes': 2,
            #     'class_names': [f'RW{i}' for i in range(4)] 
            #     },
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

                for label_key in tests[test_name]['label_key']:
                    print(label_key)

                    # parsers and shit
                    cv_parsers = {
                            _layer: partial(
                                trim_corevectors,
                                module = _layer,
                                cv_dim = cv_dim,
                                label_key = label_key
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
                                device = device,
                                label_key = label_key
                                )

                    peepholes = Peepholes(
                            path = phs_path,
                            name = phs_name+f'.{fit}.{n_cluster}.{cv_dim}.{test_name}.{emb_size}.{ci}.{label_key}',
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

                            if (driller._clas_path).exists():
                                print(driller._clas_path)
                                if (driller._empp_file).exists():
                                    print(driller._empp_file)
                                    print(f'Loading Classifier for {drill_key}') 
                                    driller.load()
                                else:
                                    print(f'Loading Classifier for {drill_key}') 
                                    driller.load_without_empp()

                                    print(f'Fitting time for {drill_key}  = ', time()-t0)

                                    driller.compute_empirical_posteriors(
                                            datasets = s,
                                            corevectors = cv,
                                            loader = tests[test_name]['empp_fit_key'],
                                            batch_size = bs,
                                            verbose=verbose
                                            )
                                    driller.save()
                                plt.imshow(driller._empp.cpu())
                                plt.savefig(driller._clas_path / f'{drill_key}_empp_{label_key}.png')
                                plt.close()

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
                                plt.savefig(driller._clas_path / f'{drill_key}_empp_{label_key}.png')
                    
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
                        
                        # for _layer in target_layers:

                        #     for a, c in enumerate(corruptions):
                                
                        #         cns = tests[test_name]['class_names']

                        #         loader_key = tests[test_name]['loaders'][1]
                                
                        #         idx = (s._dss[loader_key]['detection'] == 1) & (s._dss[loader_key]['corruption']==a) #s._dss[loader_key]['detection'] == 1  
                        #         result = ph._phs[loader_key][_layer]['peepholes'][idx]
                        #         label = s._dss[loader_key][tests[test_name]['label_key']][idx]
                                
                        #         # confusion matrix
                        #         cm = confusion_matrix(label, result.argmax(dim=1), normalize='true')
                        #         disp = ConfusionMatrixDisplay(cm, display_labels=cns)
                        #         disp.plot(cmap='Blues', colorbar=False, values_format=".2f", im_kw={'norm': norm})

                        #         # text around
                        #         for text in disp.text_.ravel():   # all the text objects (numbers in cells)
                        #             text.set_fontsize(14) 

                        #         disp.ax_.tick_params(axis='x', labelsize=14)
                        #         disp.ax_.tick_params(axis='y', labelsize=14)
                        #         disp.ax_.set_xlabel('Predicted label', fontsize=16, labelpad=10)
                        #         disp.ax_.set_ylabel('True label', fontsize=16, labelpad=10)
                                    
                        #         #disp.ax_.set_title(f'Confusion Matrix : cv_dim={cv_dim} & n_cluster={n_cluster}')
                        #         plt.tight_layout()

                        #         # save and close
                        #         plt.savefig(Path(plots_path)/f"CM.{fit}.{test_name}.{n_cluster}.{cv_dim}.{emb_size}.{ci}.{c}_only_test.png", bbox_inches='tight', dpi=300)
                        #         plt.close()

                            # disp.tick_params(axis='x', rotation=45)
                            # disp.title(f'Confusion Matrix : cv_dim={cv_dim} & n_cluster={n_cluster}')
                            # disp.tight_layout()
                            # disp.savefig(Path(plots_path)/f"CM.{fit}.{test_name}.{n_cluster}.{cv_dim}.{emb_size}.{ci}_only_test.png", bbox_inches='tight', dpi=300)
                            # disp.close()


                            # cns = tests[test_name]['class_names']

                            # # #for a, c in enumerate(corruptions):
                            # fig, axs = plt.subplots(1, 2, figsize=(20, 8))

                            # for loader_n, loader_key in enumerate(tests[test_name]['loaders']):
                            #     # the data 
                            #     idx = s._dss[loader_key]['detection'] == 1 #(s._dss[loader_key]['detection'] == 1) & (s._dss[loader_key]['corruption']==a)
                            #     result = ph._phs[loader_key][_layer]['peepholes'][idx]
                            #     label = s._dss[loader_key][tests[test_name]['label_key']][idx]
                                
                            #     # confusion matrix
                            #     cm = confusion_matrix(label, result.argmax(dim=1), normalize='true')
                            #     disp = ConfusionMatrixDisplay(cm, display_labels=cns)
                            #     disp.plot(ax=axs[loader_n], cmap='Blues', colorbar=False, values_format=".2f", im_kw={'norm': norm})

                            #     # text around
                            #     axs[loader_n].set_title(f"{loader_key.capitalize()} set")
                            #     axs[loader_n].tick_params(axis='x', rotation=45)
                            # fig.suptitle(f'Confusion Matrix : cv_dim={cv_dim} & n_cluster={n_cluster}')
                            # plt.tight_layout()
                            # plt.savefig(Path(plots_path)/f"CM.{fit}.{test_name}.{n_cluster}.{cv_dim}.{emb_size}.{ci}.png", bbox_inches='tight', dpi=300)
                            # plt.close()