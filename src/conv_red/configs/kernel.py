from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD as Reducer
from peepholelib.models.model_wrap import get_in_activations as act_parser

bs_red_scale = 1
save_input = True
save_output = False 

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {'svd_rank': _l.out_channels}
    return ret

# TODO: temp for testing
def test_configs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {
                'cv_dim': 20,
                'n_clusters': 30,
                }
    return ret
