from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD as Reducer
from peepholelib.models.model_wrap import get_in_activations as act_parser

from ray.tune import qrandint 

bs_red_scale = 2**-1
save_input = True
save_output = False 

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {'rank': 1024}
    return ret

# temp for testing
def test_configs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {
                'cv_dim': 20,
                'n_clusters': 30,
                'magnitude': 0.004,
                }
    return ret

def reduction_param_space(red_kw):
    ret = {}
    for _l, _kw in red_kw.items():
        ub = _kw['rank']
        step = (ub-50)//10
        ret[_l] = {
                'cv_dim': qrandint(50, ub, step),
                }
    return ret
