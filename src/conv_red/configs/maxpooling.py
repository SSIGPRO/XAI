from peepholelib.coreVectors.dimReduction.avgPooling import AvgPooling as Reducer 

cvs_name = 'maxpooling'

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {}
    return ret
