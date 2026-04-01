# Python stuff
from functools import partial

# Ray Stuff
from torch import linspace
from ray.tune import choice 

# Peepholelib stuff
from peepholelib.peepholes.DeepMahalanobisDistance.DMD import DeepMahalanobisDistance as Driller 
from peepholelib.scores.dmd import DMD_score as dmd_score

bs_analysis_scale = 2**-2

def get_drillers_kwargs(**kwargs):
    path = kwargs['path']
    name = kwargs['name']
    tl = kwargs['target_layers']
    nl_model = kwargs['nl_model']
    model = kwargs['model']
    configs = kwargs['configs']
    act_parser = kwargs['act_parser']
    save_input = kwargs['save_input']
    save_output = kwargs['save_output']
    device = kwargs['device']

    ret = {}
    for _l in tl:
        cv_dim = configs[_l]['cv_dim']
        mag = configs[_l]['magnitude']
        ret[_l] = {
                'path': path,
                'name': f'{name}.{_l}.{cv_dim}.{mag}', 
                'target_module': _l,
                'nl_model': nl_model,
                'n_features': cv_dim,
                'model': model,
                # divide by 1000 to avoid large file names
                'magnitude': mag/1000,
                'std_transform': [0.229, 0.224, 0.225],
                'act_parser': act_parser,
                'save_input': save_input,
                'save_output': save_output,
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        # mag is divided by 1000 at dmd to avoid large file names
        _l['magnitude'] = choice(linspace(0, 10, 10).round().numpy().tolist())
    configs['model'] = args.model 
    configs['reduction'] = args.reduction 
    configs['analysis'] = args.analysis
    return configs 

def get_score_fns(model):
    return {
            'DMD-ood': partial(
                dmd_score,
                pos_loader_train = 'CIFAR100-val-'+model,
                pos_loader_test = 'CIFAR100-test-'+model,
                neg_loaders = {
                    'Places365-test-'+model: ['Places365-val-'+model],
                    'SVHN-test-'+model: ['SVHN-val-'+model],
                    'Textures-test-'+model: ['Textures-val-'+model],
                    },
                ),
            'DMD-aa': partial(
                dmd_score,
                pos_loader_train = 'CIFAR100-val-'+model,
                pos_loader_test = 'CIFAR100-test-'+model,
                neg_loaders = {
                    'CIFAR100-test-BIM-'+model: ['CIFAR100-val-BIM-'+model],
                    'CIFAR100-test-CW-'+model: ['CIFAR100-val-CW-'+model],
                    'CIFAR100-test-DF-'+model: ['CIFAR100-val-DF-'+model],
                    'CIFAR100-test-PGD-'+model: ['CIFAR100-val-PGD-'+model],
                    },
                    ),
        }

def get_auc_kwargs_ood(model):
    return {
            'ori_loaders': {
                'DMD-ood': [
                    'Places365-val-'+model,
                    'SVHN-val-'+model,
                    'Textures-val-'+model,
                    ],
                },
            'atk_loaders': [
                'Places365-test-'+model,
                'SVHN-test-'+model,
                'Textures-test-'+model,
                ],
            'filter_key': None
            }

def get_auc_kwargs_aa(model):
    return {
            'ori_loaders': {
                'DMD-aa': [
                    'CIFAR100-val-BIM-'+model,
                    'CIFAR100-val-CW-'+model,
                    'CIFAR100-val-DF-'+model,
                    'CIFAR100-val-PGD-'+model,
                    ],
                },
            'atk_loaders': [
                'CIFAR100-test-BIM-'+model,
                'CIFAR100-test-CW-'+model,
                'CIFAR100-test-DF-'+model,
                'CIFAR100-test-PGD-'+model,
                ]
            }
