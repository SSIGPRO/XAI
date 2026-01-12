import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# python stuff
from time import time
from functools import partial

# torch stuff
import torch
import torchvision
from cuda_selector import auto_cuda


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
from peepholelib.scores.dmd import DMD_score as dmd_score 



from peepholelib.plots.confidence import plot_confidence
from peepholelib.plots.ood import plot_ood
from peepholelib.plots.calibration import plot_calibration
from peepholelib.plots.atks import auc_atks 

import random
from copy import deepcopy
from typing import Dict, List, Optional

from calculate_layer_importance import layer_importance_lolo_deltas_per_loader_okko as layer_importance, topk_layers_per_loader 


def average_random_layer_scores(*,ds, ph, target_layers_pool, scores_file, target_k=10,           
    n_runs=20, proto_key="CIFAR100-train", score_name="LACS", batch_size=128,verbose=False):

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

def avg_dmd_over_random_layer_subsets(*, target_layers: List[str], dmd_ph, pos_loader_train: str, pos_loader_test: str,
        neg_loaders: Dict[str, List[str]],score_name: str = "DMD-A", n_iters: int = 1000, subset_size: int = 10, seed: int = 0,
        verbose: bool = True, scores_file: Optional[str] = None, append_scores: Optional[dict] = None,) -> dict:
        """Compute DMD scores averaged over random subsets of layers."""

        rng = random.Random(seed)

        sums: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}

        # Optionally: keep other existing scores around, but we will overwrite score_name at the end.
        base_scores = deepcopy(append_scores) if append_scores is not None else {}

        for it in range(n_iters):
                subset = rng.sample(target_layers_pool, k=subset_size)

                iter_scores = dmd_score(
                peepholes=dmd_ph,
                pos_loader_train=pos_loader_train,
                pos_loader_test=pos_loader_test,
                neg_loaders=neg_loaders,
                append_scores={},          
                score_name=score_name,
                target_modules=subset,     
                verbose=False,
                )

                for loader_name, score_dict in iter_scores.items():
                        if score_name not in score_dict:
                                continue
                        t = score_dict[score_name]

                        if t.is_cuda:
                                t = t.detach().cpu()
                        else:
                                t = t.detach()

                        t = t.to(torch.float64)

                        if loader_name not in sums:
                                sums[loader_name] = torch.zeros_like(t, dtype=torch.float64)
                                counts[loader_name] = 0

                        sums[loader_name] += t
                        counts[loader_name] += 1

                if verbose and (it + 1) % max(1, n_iters // 10) == 0:
                        print(f"[DMD avg] iteration {it+1}/{n_iters}")

        avg_scores = deepcopy(base_scores)
        for loader_name, s in sums.items():
                avg = s / max(1, counts[loader_name])
                if loader_name not in avg_scores:
                        avg_scores[loader_name] = {}
                avg_scores[loader_name][score_name] = avg

        if scores_file is not None:
                torch.save(avg_scores, scores_file)

        return avg_scores


if __name__ == "__main__":

        use_cuda = torch.cuda.is_available()
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
       # device  = torch.device('cuda:2')
        print(f"Using {device} device")

        #--------------------------------
        # Directories definitions
        #--------------------------------
        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        ds_path = Path('/srv/newpenny/XAI/CN/mobilenet_data')

        # model parameters
        seed = 29
        bs = 512
        n_threads = 1

        model_dir = '/srv/newpenny/XAI/models'
        model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

        phs_path = Path.cwd()/'/srv/newpenny/XAI/CN/mobilenet_data/peepholes_all/peepholes_100'
        phs_name = 'peepholes'
        dmd_phs_name = 'peepavg'

        plots_path = Path.cwd()/'temp_plots/conf/mobilenet/avg'

        scores_file = Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/scores/temp_score_cifar100_avg')
        scores_file.parent.mkdir(parents=True, exist_ok=True)
        if scores_file.exists():
                scores = torch.load(scores_file)
        else:
                scores = dict()

        verbose = True 

        target_layers = [
        'features.1.conv.0.0', 'features.1.conv.1',
        'features.2.conv.0.0','features.2.conv.1.0','features.2.conv.2',
        'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2', 
        'features.4.conv.0.0', 'features.4.conv.1.0', 'features.4.conv.2',
        'features.5.conv.0.0', 'features.5.conv.1.0', 'features.5.conv.2',
        'features.6.conv.0.0','features.6.conv.1.0', 'features.6.conv.2',
        'features.7.conv.0.0', 'features.7.conv.1.0','features.7.conv.2',
        'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2',
        'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2',  
        'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2',
        'features.11.conv.0.0', 'features.11.conv.1.0', 'features.11.conv.2',
        'features.12.conv.0.0', 'features.12.conv.1.0',  'features.12.conv.2',
        'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2',
        'features.14.conv.0.0', 'features.14.conv.1.0', 'features.14.conv.2',
        'features.15.conv.0.0', 
        'features.15.conv.1.0', 'features.15.conv.2',
        'features.16.conv.0.0', 'features.16.conv.1.0', 'features.16.conv.2', 
        'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2',
        'features.18.0', 
        'classifier.1',
        ]

        # # #worst ones
        # target_layers = ['features.1.conv.0.0','features.4.conv.1.0','features.4.conv.2', 'features.7.conv.2','features.8.conv.0.0',
        #         'features.8.conv.2','features.9.conv.0.0', 'features.11.conv.1.0',
        #         'features.12.conv.1.0','features.13.conv.2' ]

        # # best auc
        # target_layers = ['features.1.conv.1','features.2.conv.0.0','features.4.conv.1.0','features.9.conv.1.0','features.5.conv.1.0' ,'features.15.conv.0.0', 
        # 'features.17.conv.1.0','features.17.conv.2', 'features.18.0', 'classifier.1']

        
        # #best fr95
        # target_layers=[
        #         'features.14.conv.2', 'features.17.conv.0.0', 'features.17.conv.2', 'features.15.conv.2', 'features.11.conv.2','features.17.conv.1.0', 
        #         'features.14.conv.1.0','features.15.conv.0.0','features.18.0', 'classifier.1'
        # ]

        # best coverage (threshold =0.95)
        # target_layers = ['features.2.conv.0.0','features.3.conv.0.0','features.3.conv.1.0','features.3.conv.2','features.5.conv.1.0',
        # 'features.6.conv.1.0','features.8.conv.1.0','features.9.conv.1.0','features.17.conv.2','classifier.1']

        # best coverage (threshold =0.7-0.89)
        # target_layers = ['features.2.conv.0.0','features.3.conv.2','features.5.conv.1.0','features.6.conv.1.0','features.8.conv.1.0',
        # 'features.9.conv.1.0','features.17.conv.1.0','features.17.conv.2','features.18.0','classifier.1']

        # best coverage (threshold =0.6)
        # target_layers = ['features.2.conv.0.0','features.3.conv.2','features.6.conv.1.0','features.8.conv.1.0','features.9.conv.1.0',
        # 'features.14.conv.2','features.17.conv.1.0','features.17.conv.2','features.18.0','classifier.1']
        
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
        # 'SVHN-val',
        # 'SVHN-test',
        # 'Places365-val',
        # 'Places365-test',
        ]

        #--------------------------------
        # Model 
        #--------------------------------
        nn = torchvision.models.mobilenet_v2(pretrained=True)

        n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

        model = ModelWrap(
                model = nn,
                device = device
                )
                                                
        model.update_output(
                output_layer = 'classifier.1', 
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
                
        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                )
        # dmd_peepholes = Peepholes(
        #     path = phs_path,
        #     name = dmd_phs_name,
        #     )

        with datasets as ds, peepholes as ph: 
                ds.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        ) 

                ph.load_only(
                        loaders = loaders,
                        verbose = verbose 
                        )
                # dmd_ph.load_only(
                #         loaders = loaders,
                #         verbose = verbose 
                #         )
 
                # deltas = layer_importance(score_fn=proto_score,
                #         datasets=ds, peepholes=peepholes,
                #         target_modules=target_layers, loaders=loaders,
                #         score_name="LACS", proto_key="CIFAR100-train",
                #         batch_size=bs,
                #         append_scores=scores, verbose=True,
                #         )
                # topk = topk_layers_per_loader(deltas, k=30,
                #         mode="fpr95",     # or "fpr95" or "joint"
                #         )
                # quit()

                if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'LACS' in scores['CIFAR100-test'])): 
                #get scores
                        # scores, protoclasses = proto_score(
                        #         datasets = ds,
                        #         peepholes = ph,
                        #         proto_key = 'CIFAR100-train',
                        #         score_name = 'LACS',
                        #         batch_size = bs, 
                        #         target_modules = target_layers,
                        #         append_scores = scores,
                        #         verbose = verbose,
                        #         )
                                
                        #torch.save(scores, scores_file)
                        scores_avg = average_random_layer_scores(
                                ds=ds,
                                ph=ph,
                                target_layers_pool=target_layers,
                                target_k=10,
                                batch_size=bs,
                                verbose=verbose,
                                scores_file=scores_file,
                        )
                        scores = scores_avg
                        print(scores_avg)

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
                #         print('mconf scores found')~

                #if (not 'Places365-test' in scores) or (('Places365-test' in scores) and (not 'DMD-U' in scores['Places365-test'])): 
                #         scores = dmd_score(
                #         peepholes = dmd_ph,
                #         pos_loader_train = 'CIFAR100-val',
                #         pos_loader_test = 'CIFAR100-test',
                #         neg_loaders = {
                #                 'Places365-test': ['SVHN-val'],
                #                 'SVHN-test': ['Places365-val']
                #                 },
                #         append_scores = scores,
                #         score_name = 'DMD-U'
                #         )
                #         torch.save(scores, scores_file)

                #         scores = avg_dmd_over_random_layer_subsets(target_layers=target_layers, dmd_ph=dmd_ph,
                #                 pos_loader_train='CIFAR100-val', pos_loader_test='CIFAR100-test',
                #                 neg_loaders = {
                #                 'CIFAR100-test': ['CIFAR100-val'],
                #                 'CIFAR100-C-test-c0': ['CIFAR100-C-val-c0'],
                #                 'CIFAR100-C-test-c1': ['CIFAR100-C-val-c1'],
                #                 'CIFAR100-C-test-c2': ['CIFAR100-C-val-c2'],
                #                 'CIFAR100-C-test-c3': ['CIFAR100-C-val-c3'],
                #                 'CIFAR100-C-test-c4': ['CIFAR100-C-val-c4'],
                #                 'Places365-test': ['Places365-val'],
                #                 'SVHN-test': ['SVHN-val']
                #                 },
                #                 score_name='DMD-A', 
                #                 scores_file=scores_file,  append_scores=scores,        
                #                 )
                # else:
                #         print('dmd-u scores found')

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
                #         #'MSP': 'CIFAR100-test',
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
                #         #'MSP': 'CIFAR100-test',
                #         },
                #         atk_loaders = ['BIM-CIFAR100-test', 'CW-CIFAR100-test', 'DF-CIFAR100-test', 'PGD-CIFAR100-test'],
                #         verbose = verbose
                #         )



