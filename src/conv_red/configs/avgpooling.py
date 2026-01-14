from peepholelib.coreVectors.dimReduction.avgPooling import AvgPooling as Reducer 
from peepholelib.models.model_wrap import get_out_activations as act_parser

bs_red_scale = 1
save_input = False 
save_output = True 

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {}
    return ret

# TODO: temp for testing
def test_configs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {
                'cv_dim': _l.out_channels,
                'n_clusters': 30,
                }
    return ret
