from peepholelib.coreVectors.dimReduction.svds.conv2d_avg_kernel_svd import Conv2dAvgKernelSVD as Reducer

cvs_name = 'kernel'

# function to compute reducers's extra kwargs
def get_reducer_kwargs(tl):
    ret = {}
    for _n, _l in tl.items():
        ret[_n] = {'svd_rank': _l.out_channels}
    return ret
