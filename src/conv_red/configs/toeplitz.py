from peepholelib.coreVectors.dimReduction.svds.conv2d_toeplitz_svd import Conv2dToeplitzSVD as Reducer

cvs_name = 'toeplitz'

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {'svd_rank': 1024}
    return ret
