# Python stuff
from functools import partial

# Ray Stuff
from ray.tune import quniform 

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
                'magnitude': mag, #0.004
                'std_transform': [0.229, 0.224, 0.225],
                'act_parser': act_parser,
                'save_input': save_input,
                'save_output': save_output,
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        _l['magnitude'] = quniform(0, 1e-1, 1e-2)
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
                    'CIFAR100-C-test-c4-'+model: ['CIFAR100-C-val-c4-'+model],
                    'Places365-test-'+model: ['Places365-val-'+model],
                    'SVHN-test-'+model: ['SVHN-val-'+model]
                    },
                ),
            'DMD-aa': partial(
                dmd_score,
                pos_loader_train = 'CIFAR100-val-'+model,
                pos_loader_test = 'CIFAR100-test-'+model,
                neg_loaders = {
                    'CIFAR100-test-BIM-'+model: ['CIFAR100-val-BIM-'+model],
                    'CIFAR100-test-CW-'+model: ['CIFAR100-val-CW-'+model],
                    },
                    ),
        }

def get_auc_kwargs_ood(model):
    return {
            'ori_loaders': {
                'DMD-ood': ['CIFAR100-C-val-c4-'+model, 'Places365-val-'+model, 'SVHN-val-'+model],
                },
            'atk_loaders': ['CIFAR100-C-test-c4-'+model, 'Places365-test-'+model, 'SVHN-test-'+model],
            'filter_key': None
            }

def get_auc_kwargs_aa(model):
    return {
            'ori_loaders': {
                'DMD-aa': ['CIFAR100-val-BIM-'+model, 'CIFAR100-val-CW-'+model],
                },
            'atk_loaders': ['CIFAR100-test-BIM-'+model, 'CIFAR100-test-CW-'+model]
            }