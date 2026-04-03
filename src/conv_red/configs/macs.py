# Python stuff
from functools import partial

# Ray Stuff
from ray.tune import qrandint 

# Peepholelib stuff
from peepholelib.peepholes.classifiers.tgmm import GMM as Driller 
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score 

bs_analysis_scale = 2**5

def get_drillers_kwargs(**kwargs):
    path = kwargs['path']
    name = kwargs['name']
    tl = kwargs['target_layers']
    nl_model = kwargs['nl_model']
    configs = kwargs['configs']
    device = kwargs['device']

    ret = {}
    for _l in tl:
        cv_dim = configs[_l]['cv_dim']
        n_clusters = configs[_l]['n_clusters']
        ret[_l] = {
                'path': path,
                'name': f'{name}.{_l}.{cv_dim}.{n_clusters}',
                'target_module': _l,
                'nl_classifier': n_clusters,
                'nl_model': nl_model,
                'n_features': cv_dim,
                'cls_kwargs': {
                    'covariance_regularization': 1e-4,
                    'convergence_tolerance': 1e-2
                    },
                'device': device
                } 
    return ret

def analysis_param_space(configs, args):
    for _n, _l in configs.items():
        _l['n_clusters'] = qrandint(50, 500, 50)
    configs['model'] = args.model 
    configs['reduction'] = args.reduction 
    configs['analysis'] = args.analysis
    return configs 

score_fns = {
        'MACS': partial(
            proto_score,
            proto_key = 'CIFAR100-train'
            )
        }

auc_kwargs_ood = {
        'ori_loaders': {
            'MACS': 'CIFAR100-test',
            },
        'atk_loaders': ['CIFAR100-C-test-c4', 'Places365-test', 'SVHN-test'],
        'filter_key': None
        }

auc_kwargs_aa = {
        'ori_loaders': {
            'MACS': 'CIFAR100-test',
            },
        'atk_loaders': ['BIM-CIFAR100-test', 'CW-CIFAR100-test']
        }
