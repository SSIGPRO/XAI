# Python stuff
from functools import partial

# Our stuff
from peepholelib.peepholes.classifiers.tgmm import GMM as Driller 

bs_analysis_scale = 1

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
