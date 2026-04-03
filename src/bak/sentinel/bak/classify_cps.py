import torch

def classify_cps(**kwargs):
    dss = kwargs['datasets']
    phs = kwargs['peepholes']
    loaders = kwargs.get('loaders')
    target_modules = kwargs.get('target_modules')
    proto_key = kwargs.get('proto_key')
    proto_th = kwargs.get('proto_threshold', 0.5)
    verbose = kwargs.get('verbose', True)
    
    cpss = phs.get_conceptograms(loaders=loaders, target_modules=target_modules, verbose=verbose)

    # sizes and values just to facilitate 
    nd = cpss[loaders[0]].shape[1] # number of layers (distributions)
    nc = cpss[loaders[0]].shape[2] # number of classes
    
    cps = cpss[proto_key]
    confs = (dss._dss[proto_key]['output'] - dss._dss[proto_key]['data']).squeeze().norm(dim=(1, 2))
    # compute proto-classes
    proto = torch.zeros(nc, nd, nc)
    for i in range(nc):
        idx = confs < proto_th
        _p = cps[idx].sum(dim=0)  ## P'_j
        _p /= _p.sum(dim=1, keepdim=True)
        proto[i][:] = _p[:]
    
    # compute protoclass score
    cls = {}
    for ds_key in loaders:
        cps = cpss[ds_key]
        ns = cps.shape[0] # number of samples 
        scores = torch.zeros(ns, nc)
        for i in range(nc):
            pi = proto[i]
            cp = cps
            scores[:, i] = (pi*cp).sum(dim=(1, 2))#/(pi.norm()*cp.norm(dim=(1, 2)))
        
        cls[ds_key] = scores.max(axis=1).indices
    return cls 
