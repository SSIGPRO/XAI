from utils.get_best_configs import get_best_config

def test_configs(tl, file):
    ret = {}
    bc = get_best_config(file)
    # tunned convolution configs
    for _n, _l in tl.items():
        ret[_n] = {}
        for _c in ['cv_dim', 'n_clusters', 'magnitude']:
            if _n+'/'+_c in bc:
                ret[_n][_c] = bc[_n+'/'+_c] 

        # avg_pooling with MACS case
        if _n+'/cv_dim' not in bc:
            ret[_n]['cv_dim'] = _l.out_channels 

    return ret
