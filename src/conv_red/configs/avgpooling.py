from peepholelib.coreVectors.dimReduction.avgPooling import AvgPooling as Reducer 
from peepholelib.models.model_wrap import get_out_activations as act_parser

bs_red_scale = 1
save_input = False 
save_output = True 

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {'fs': _l.out_channels}
    return ret

# temp for testing
def test_configs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {
                'cv_dim': _l.out_channels,
                'n_clusters': 30,
                'magnitude': 0.004,
                }
    return ret

def reduction_param_space(red_kw):
    ret = {}
    for _l, _kw in red_kw.items():
        ret[_l] = {
                'cv_dim': _kw['fs'],
                }
    return ret
