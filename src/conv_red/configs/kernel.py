from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD as Reducer
from peepholelib.models.model_wrap import get_in_activations as act_parser

from torch import linspace, int32
from ray.tune import choice 

bs_red_scale = 1
save_input = True
save_output = False 

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {'rank': min(_l.out_channels, _l.in_channels)}
    return ret

def reduction_param_space(red_kw):
    ret = {}
    for _l, _kw in red_kw.items():
        ub = _kw['rank']
        ret[_l] = {
                'cv_dim': choice(linspace(50, ub, 10, dtype=int32).numpy().tolist()),
                }
    return ret
