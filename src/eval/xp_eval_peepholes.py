import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial

# Our stuff
import peepholelib
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.peepholes.parsers import get_images  
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as DMD 

from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
from peepholelib.scores.model_confidence import model_confidence_score as mconf_score 
from peepholelib.scores.doctor import DOCTOR_score as doctor_score 
from peepholelib.scores.relu import RelU_score as relu_score 
from peepholelib.featureSqueezing.FeatureSqueezingDetector import FeatureSqueezingDetector as FSD
from peepholelib.featureSqueezing.preprocessing import NLM_filtering_torch, NLM_filtering_cv, bit_depth_torch, MedianPool2d
from peepholelib.scores.feature_squeezing import feature_squeezing_score as fs_score 
from peepholelib.scores.dmd import DMD_score as dmd_score 

from peepholelib.plots.confidence import plot_confidence
from peepholelib.plots.ood import plot_ood
from peepholelib.plots.calibration import plot_calibration
from peepholelib.plots.atks import auc_atks 

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":
    
    scores_file = Path('./temp_scores/'+sys.argv[1])
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    if scores_file.exists():
        scores = torch.load(scores_file)
    else:
        scores = dict()
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = Model()
    model = ModelWrap(
            model = nn,
            device = device
            )
    
    model.update_output(
            output_layer = output_layer, 
            to_n_classes = 100,
            overwrite = True 
            )
    
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    model.set_target_modules(target_modules=target_layers, verbose=verbose)
    
    # feature squeezing stuff
    fsd = FSD(
            model = model,
            prepro_dict = {
                'median': MedianPool2d(kernel_size=3, stride=1, padding=1),
                'bit_depth': partial(bit_depth_torch, bits=5),
                'nlm': partial(NLM_filtering_torch, kernel_size=11, std=4.0, kernel_size_mean=3, sub_filter_size=32),
                }
            )

    # Peepholes
    datasets = ParsedDataset(
            path = ds_path,
            )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            )

    dmd_peepholes = Peepholes(
            path = phs_path,
            name = dmd_phs_name,
            )

    with datasets as ds, peepholes as ph, dmd_peepholes as dmd_ph: 
        
        ds.load_only(
                loaders = loaders,
                verbose = verbose 
                ) 

        ph.load_only(
                loaders = loaders,
                verbose = verbose 
                )
            
        dmd_ph.load_only(
                loaders = loaders,
                verbose = verbose 
                )
        
        if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'LACS' in scores['CIFAR100-test'])): 
            # get scores
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
        else: 
            print('proto scores found')

        if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'MSP' in scores['CIFAR100-test'])): 
            scores = mconf_score(
                    datasets = ds,
                    batch_size = bs, 
                    append_scores = scores,
                    verbose = verbose
                    ) 
            torch.save(scores, scores_file)
        else:
            print('mconf scores found')

        if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'DOC' in scores['CIFAR100-test'])): 
            scores = doctor_score(
                    datasets = ds,
                    model = model,
                    batch_size = bs, 
                    score_name = 'DOC',
                    append_scores = scores,
                    verbose = verbose
                    )
            torch.save(scores, scores_file)
        else:
            print('doc scores found')


        if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'Rel-U' in scores['CIFAR100-test'])): 
            scores = relu_score(
                    datasets = ds,
                    fit_key = 'CIFAR100-train',
                    batch_size = bs, 
                    append_scores = scores,
                    verbose = verbose
                    )
            torch.save(scores, scores_file)
        else:
            print('relu scores found')

        if (not 'CIFAR100-test' in scores) or (('CIFAR100-test' in scores) and (not 'FS' in scores['CIFAR100-test'])): 
            scores = fs_score(
                    datasets = ds,
                    detector = fsd,
                    batch_size = 2**6, 
                    append_scores = scores,
                    score_name = 'FS',
                    verbose = verbose
                    ) 
            torch.save(scores, scores_file)
        else:
            print('fs scores found')

        print('\n----------------------\n  Conf \n----------------------\n')
        # make plots
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

        if (not 'CIFAR100-C-test-c0' in scores) or (('CIFAR100-C-test-c0' in scores) and (not 'DMD-A' in scores['CIFAR100-C-test-c0'])): 
            # DMD Aware
            scores = dmd_score(
                    peepholes = dmd_ph,
                    pos_loader_train = 'CIFAR100-val',
                    pos_loader_test = 'CIFAR100-test',
                    neg_loaders = {
                        'CIFAR100-C-test-c0': ['CIFAR100-C-val-c0'],
                        'CIFAR100-C-test-c1': ['CIFAR100-C-val-c1'],
                        'CIFAR100-C-test-c2': ['CIFAR100-C-val-c2'],
                        'CIFAR100-C-test-c3': ['CIFAR100-C-val-c3'],
                        'CIFAR100-C-test-c4': ['CIFAR100-C-val-c4'],
                        'Places365-test': ['Places365-val'],
                        'SVHN-test': ['SVHN-val']
                        },
                    append_scores = scores,
                    score_name = 'DMD-A'
                    )
            torch.save(scores, scores_file)
        else:
            print('dmd-a scores found')

        print('\n----------------------\n  OOD Near \n----------------------\n')
        plot_ood(
                scores = scores,
                path = plots_path,
                id_loaders = {
                    'LACS': 'CIFAR100-test',
                    'MSP': 'CIFAR100-test',
                    'DOC': 'CIFAR100-test',
                    'Rel-U': 'CIFAR100-test',
                    'FS': 'CIFAR100-test',
                    'DMD-A': [f'CIFAR100-C-val-c{i}' for i in range(5)],
                    },
                ood_loaders = [f'CIFAR100-C-test-c{i}' for i in range(5)],
                suffix = 'Corruption',
                loaders_renames = [f'c{i}' for i in range(5)],
                verbose = verbose
                ) 

        # DMD Unaware
        if (not 'Places365-test' in scores) or (('Places365-test' in scores) and (not 'DMD-U' in scores['Places365-test'])): 
            scores = dmd_score(
                    peepholes = dmd_ph,
                    pos_loader_train = 'CIFAR100-val',
                    pos_loader_test = 'CIFAR100-test',
                    neg_loaders = {
                        'Places365-test': ['SVHN-val'],
                        'SVHN-test': ['Places365-val']
                        },
                    append_scores = scores,
                    score_name = 'DMD-U'
                    )
            torch.save(scores, scores_file)
        else:
            print('dmd-u scores found')

        print('\n----------------------\n  OOD Far \n----------------------\n')
        plot_ood(
                scores = scores,
                path = plots_path,
                id_loaders = {
                    'LACS': 'CIFAR100-test',
                    'MSP': 'CIFAR100-test',
                    'DOC': 'CIFAR100-test',
                    'Rel-U': 'CIFAR100-test',
                    'FS': 'CIFAR100-test',
                    'DMD-A': ['SVHN-val', 'Places365-val'],
                    'DMD-U': ['Places365-val', 'SVHN-val'],
                    },
                ood_loaders = ['SVHN-test', 'Places365-test'],
                suffix = 'Far',
                verbose = verbose
                ) 

        print('\n----------------------\n  ATKS \n----------------------\n')

        if (not 'BIM-CIFAR100-test' in scores) or (('BIM-CIFAR100-test' in scores) and (not 'DMD-A-atks' in scores['BIM-CIFAR100-test'])): 
            scores = dmd_score(
                    peepholes = dmd_ph,
                    pos_loader_train = 'CIFAR100-val',
                    pos_loader_test = 'CIFAR100-test',
                    neg_loaders = {
                        'BIM-CIFAR100-test': ['BIM-CIFAR100-val'],
                        'CW-CIFAR100-test': ['CW-CIFAR100-val'],
                        'DF-CIFAR100-test': ['DF-CIFAR100-val'],
                        'PGD-CIFAR100-test': ['PGD-CIFAR100-val'],
                        },
                    append_scores = scores,
                    score_name = 'DMD-A-atks'
                    )
            torch.save(scores, scores_file)
        else:
            print('dmd-A-atks scores found')

        if (not 'BIM-CIFAR100-test' in scores) or (('BIM-CIFAR100-test' in scores) and (not 'DMD-U-atks' in scores['BIM-CIFAR100-test'])): 
            scores = dmd_score(
                    peepholes = dmd_ph,
                    pos_loader_train = 'CIFAR100-val',
                    pos_loader_test = 'CIFAR100-test',
                    neg_loaders = {
                        'BIM-CIFAR100-test': ['CW-CIFAR100-val', 'DF-CIFAR100-val', 'PGD-CIFAR100-val'],
                        'CW-CIFAR100-test': ['DF-CIFAR100-val', 'PGD-CIFAR100-val', 'BIM-CIFAR100-val'],
                        'DF-CIFAR100-test': ['PGD-CIFAR100-val', 'BIM-CIFAR100-val', 'CW-CIFAR100-val'],
                        'PGD-CIFAR100-test': ['BIM-CIFAR100-val', 'CW-CIFAR100-val', 'DF-CIFAR100-val'],
                        },
                    append_scores = scores,
                    score_name = 'DMD-U-atks'
                    )
            torch.save(scores, scores_file)
        else:
            print('dmd-U-atks scores found')

        auc_atks(
                datasets = ds,
                scores = scores,
                ori_loaders = {
                    'LACS': 'CIFAR100-test',
                    'MSP': 'CIFAR100-test',
                    'DOC': 'CIFAR100-test',
                    'Rel-U': 'CIFAR100-test',
                    'FS': 'CIFAR100-test',
                    'DMD-A-atks': ['BIM-CIFAR100-val', 'CW-CIFAR100-val', 'DF-CIFAR100-val', 'PGD-CIFAR100-val'],
                    'DMD-U-atks': ['CW-CIFAR100-val', 'DF-CIFAR100-val', 'PGD-CIFAR100-val', 'BIM-CIFAR100-val'],
                    },
                atk_loaders = ['BIM-CIFAR100-test', 'CW-CIFAR100-test', 'DF-CIFAR100-test', 'PGD-CIFAR100-test'],
                verbose = verbose
                )


