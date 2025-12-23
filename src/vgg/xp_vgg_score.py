import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# torch stuff
import torch
import torchvision
from cuda_selector import auto_cuda
from torchvision.models import vgg16


###### Our stuff

# Model
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd

# datasets
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.parsers import from_dataset
from peepholelib.datasets.functional.samplers import random_subsampling 
from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector as FSD
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, NLM_filtering_cv, bit_depth_torch, MedianPool2d

# peepholes
from peepholelib.peepholes.peepholes import Peepholes

#scores
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
from peepholelib.scores.model_confidence import model_confidence_score as mconf_score 
from peepholelib.scores.doctor import DOCTOR_score as doctor_score 
from peepholelib.scores.relu import RelU_score as relu_score 
from peepholelib.scores.feature_squeezing import feature_squeezing_score as fs_score 



from peepholelib.plots.confidence import plot_confidence
from peepholelib.plots.ood import plot_ood
from peepholelib.plots.calibration import plot_calibration
from peepholelib.plots.atks import auc_atks 

import random


def average_random_layer_scores(*,ds, ph, target_layers_pool, scores_file, target_k=10,           
    n_runs=1000, proto_key="CIFAR100-train", score_name="LACS", batch_size=128,verbose=False):

    running_sum = None  
    n_seen = 0

    for r in range(n_runs):
        sampled_layers = random.sample(target_layers_pool, k=target_k)

        # no append_scores here, we want only this run's scores
        scores_run, _ = proto_score(
            datasets=ds,
            peepholes=ph,
            proto_key=proto_key,
            score_name=score_name,
            batch_size=batch_size,
            target_modules=sampled_layers,
            append_scores=None,
            verbose=verbose,
        )

        # scores_run is scores_run[ds_key][score_name] = tensor(N,)
        if running_sum is None:
            running_sum = {}
            for ds_key in scores_run.keys():
                t = scores_run[ds_key][score_name].detach().to("cpu").float()
                running_sum[ds_key] = t.clone()
        else:
            for ds_key in scores_run.keys():
                running_sum[ds_key] += scores_run[ds_key][score_name].detach().to("cpu").float()

        n_seen += 1
        
    scores_avg = {ds_key: {score_name: running_sum[ds_key] / n_seen} for ds_key in running_sum.keys()}

    torch.save(scores_avg, scores_file)
    return scores_avg


if __name__ == "__main__":

        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        ds_path = Path('/srv/newpenny/XAI/CN/vgg_data/cifar100')

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
        
        svds_path = '/srv/newpenny/XAI/CN/vgg_data'
        svds_name = 'svds' 
        
        cvs_path = Path.cwd()/'/srv/newpenny/XAI/CN/vgg_data/corevectors'
        cvs_name = 'corevectors'

        drill_path = Path.cwd()/'/srv/newpenny/XAI/CN/vgg_data/drillers_all/drillers_100'
        drill_name = 'classifier'

        phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/vgg_data/peepholes_all/peepholes_100'
        phs_name = 'peepholes'

        plots_path = Path.cwd()/'temp_plots/conf/vgg/all'


        scores_file = Path('/home/claranunesbarrancos/repos/XAI/src/vgg/scores/temp_score_cifar100_all')
        scores_file.parent.mkdir(parents=True, exist_ok=True)
        if scores_file.exists():
                scores = torch.load(scores_file)
        else:
                scores = dict()

        verbose = True 

        loaders = [
        'CIFAR100-train',
        'CIFAR100-val',
        'CIFAR100-test',
        # 'CIFAR100-C-val-c0',
        # 'CIFAR100-C-test-c0',
        # 'CIFAR100-C-val-c1',
        # 'CIFAR100-C-test-c1',
        # 'CIFAR100-C-val-c2',
        # 'CIFAR100-C-test-c2',
        # 'CIFAR100-C-val-c3',
        # 'CIFAR100-C-test-c3',
        # 'CIFAR100-C-val-c4',
        # 'CIFAR100-C-test-c4',
        # 'CW-CIFAR100-val',
        # 'CW-CIFAR100-test',
        # 'BIM-CIFAR100-val',
        # 'BIM-CIFAR100-test',
        # 'DF-CIFAR100-val',
        # 'DF-CIFAR100-test',
        # 'PGD-CIFAR100-val',
        # 'PGD-CIFAR100-test',
        ]

        #--------------------------------
        # Model 
        #--------------------------------
        nn = vgg16()
        target_layers = [ 'features.0', 'features.2', 'features.5','features.7', 'features.10', 'features.12', 'features.14', 'features.17', 'features.19', 'features.21',
                            'features.24','features.26','features.28','classifier.0','classifier.3', 
                            'classifier.6',
                ]

        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

        model = ModelWrap(
                model = nn,
                device = device
                )
                                                
        model.update_output(
                output_layer = 'classifier.6', 
                to_n_classes = n_classes,
                overwrite = True 
                )
                                                
        model.load_checkpoint(
                name = model_name,
                path = model_dir,
                verbose = verbose
                )
                                                
        model.set_target_modules(
                target_modules = target_layers,
                verbose = verbose
                )

        datasets = ParsedDataset(
                path = ds_path,
                )

        # feature squeezing stuff
        fsd = FSD(
                model = model,
                prepro_dict = {
                        'median': MedianPool2d(kernel_size=3, stride=1, padding=1),
                        'bit_depth': partial(bit_depth_torch, bits=5),
                        'nlm': partial(NLM_filtering_torch, kernel_size=11, std=4.0, kernel_size_mean=3, sub_filter_size=32),
                        }
                )
        datasets = ParsedDataset(
                path = ds_path,
                )

    # Peepholes
        # for layer in target_layers:
        #         if layer in {"features.3.conv.2", "features.11.conv.2"}:
        #                 phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_50'
        #                 print(f"Its {layer}, switching to 50 clusters")
        #         elif layer == "features.5.conv.1.0":
        #                 phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_150'
        #                 print(f"Its {layer}, switching to 150 clusters")
        #         else:
        #                 print(f"its layer {layer}, so remaining with 100 clusters")
        #                 phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_100'
        #         print("phs path:", phs_path)
                
        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                )

        with datasets as ds, peepholes as ph: 
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 

                ph.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        )

                if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'LACS' in scores['CIFAR100-test'])): 
                #get scores
                        scores, protoclasses = proto_score(
                                datasets = ds,
                                peepholes = ph,
                                proto_key = 'CIFAR100-train',
                                score_name = 'LACS',
                                batch_size = bs, 
                                target_modules = target_layers,
                                append_scores = scores,
                                verbose = verbose,
                                )
                                
                        torch.save(scores, scores_file)
                        # scores_avg = average_random_layer_scores(
                        #         ds=ds,
                        #         ph=ph,
                        #         target_layers_pool=target_layers,
                        #         target_k=10,
                        #         n_runs=1000,
                        #         batch_size=bs,
                        #         verbose=verbose,
                        #         scores_file=scores_file,
                        # )
                        # scores = scores_avg
                        # print(scores_avg)

                else: 
                        print('proto scores found')
        
                # if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'MSP' in scores['CIFAR100-test'])): 
                #         scores = mconf_score(
                #                 datasets = ds,
                #                 batch_size = bs, 
                #                 append_scores = scores,
                #                 verbose = verbose
                #                 ) 
                #         torch.save(scores, scores_file)
                # else:
                #         print('mconf scores found')


                print('\n----------------------\n  Conf \n----------------------\n')
                # make plots
        
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 
                plot_confidence(
                        datasets = ds,
                        scores = scores,
                        loaders = ['CIFAR100-test'],
                        max_score = 1.,
                        path = plots_path,
                        verbose = verbose
                        )

                print('\n----------------------\n  Calib \n----------------------\n')
                plot_calibration(
                        datasets = ds,
                        scores = scores,
                        loaders = ['CIFAR100-test'],
                        calib_bin = 0.1,
                        path = plots_path,
                        verbose = verbose
                        )

                # print('\n----------------------\n  OOD Near \n----------------------\n')
                # plot_ood(
                #         scores = scores,
                #         path = plots_path,
                #         id_loaders = {
                #         'LACS': 'CIFAR100-test',
                #        # 'MSP': 'CIFAR100-test',
                #         },
                #         ood_loaders = [f'CIFAR100-C-test-c{i}' for i in range(5)],
                #         suffix = 'Corruption',
                #         loaders_renames = [f'c{i}' for i in range(5)],
                #         verbose = verbose
                #         ) 

                # auc_atks(
                #         datasets = ds,
                #         scores = scores,
                #         ori_loaders = {
                #         'LACS': 'CIFAR100-test',
                #        # 'MSP': 'CIFAR100-test',
                #         },
                #         atk_loaders = ['BIM-CIFAR100-test', 'CW-CIFAR100-test', 'DF-CIFAR100-test', 'PGD-CIFAR100-test'],
                #         verbose = verbose
                #         )




















































