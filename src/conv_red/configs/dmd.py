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

score_fns = {
        'DMD-ood': partial(
            dmd_score,
            pos_loader_train = 'CIFAR100-val',
            pos_loader_test = 'CIFAR100-test',
            neg_loaders = {
                'CIFAR100-C-test-c4': ['CIFAR100-C-val-c4'],
                'Places365-test': ['Places365-val'],
                'SVHN-test': ['SVHN-val']
                },


            ),
        'DMD-aa': partial(
            dmd_score,
            pos_loader_train = 'CIFAR100-val',
            pos_loader_test = 'CIFAR100-test',
            neg_loaders = {
                'BIM-CIFAR100-test': ['BIM-CIFAR100-val'],
                'CW-CIFAR100-test': ['CW-CIFAR100-val'],
                },
            ),
        }

auc_kwargs_ood = {
        'ori_loaders': {
            'DMD-ood': ['CIFAR100-C-val-c4', 'Places365-val', 'SVHN-val'],
            },
        'atk_loaders': ['CIFAR100-C-test-c4', 'Places365-test', 'SVHN-test'],
        'filter_key': None
        }

auc_kwargs_aa = {
        'ori_loaders': {
            'DMD-aa': ['BIM-CIFAR100-val', 'CW-CIFAR100-val'],
            },
        'atk_loaders': ['BIM-CIFAR100-test', 'CW-CIFAR100-test']
        }