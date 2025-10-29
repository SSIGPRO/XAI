import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
from matplotlib import pyplot as plt
import json

# Our stuff
import peepholelib
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.scores.protoclass import conceptogram_protoclass_score as proto_score
from peepholelib.plots.conceptograms import plot_conceptogram

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
elif sys.argv[1] == 'vgg_imagenet':
    from config_imagenet_vgg16 import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":

    scores_file = Path('./temp_scores/'+sys.argv[1])
    scores = torch.load(scores_file)
    n_cps = 5 
    #--------------------------------
    # Dataset 
    #--------------------------------

    datasets = ParsedDataset(
            path = ds_path,
            )
                                                    
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            )
    
    ticks = [f'l.{i}' for i, _ in enumerate(target_layers)]
    
    with open(Path.cwd()/f'../../data/imagenet_class_index.json', "r", encoding="utf-8") as f:
        wnid_to_idx_label = json.load(f) 
    class_names = [v[1] for k, v in sorted(wnid_to_idx_label.items(), key=lambda kv: int(kv[0]))]

    with datasets as ds, peepholes as ph: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose 
                ) 

        ph.load_only(
                loaders = loaders, 
                verbose = verbose 
                )
        
            
        # get scores
        _, protoclasses = proto_score(
                datasets = ds,
                peepholes = ph,
                proto_key = 'CIFAR100-train',
                score_name = 'LACS',
                batch_size = bs, 
                target_modules = target_layers,
                append_scores = scores,
                verbose = verbose,
                )
        
        '''
        # selects assorted samples for each range of scores.
        s = scores['CIFAR100-test']['LACS']
        # low proto score
        idx_ll = torch.logical_and(s > 0.0, s<=0.25).nonzero().squeeze()
        _ridx = torch.randperm(len(idx_ll))[:n_cps]
        idx_ll = idx_ll[_ridx]

        idx_ml = torch.logical_and(s > 0.25, s<=0.5).nonzero().squeeze()
        _ridx = torch.randperm(len(idx_ml))[:n_cps]
        idx_ml = idx_ml[_ridx]

        idx_mh = torch.logical_and(s > 0.50, s<=0.75).nonzero().squeeze()
        _ridx = torch.randperm(len(idx_mh))[:n_cps]
        idx_mh = idx_mh[_ridx]

        idx_hh = torch.logical_and(s > 0.75, s<=1.0).nonzero().squeeze()
        _ridx = torch.randperm(len(idx_hh))[:n_cps]
        idx_hh = idx_hh[_ridx]

        idx = torch.cat((idx_ll, idx_ml, idx_mh, idx_hh))
        '''
        
        # we picked these ones from the random generated ones
        idx = [131, 2686]#, 405, 3594, 5000, 9617]

        cifar_path = '/srv/newpenny/dataset/CIFAR100'
        plot_conceptogram(
                path = plots_path,
                name = 'conceptogram',
                datasets = ds,
                peepholes = ph,
                #loaders = ['CIFAR100-test'], # these with the protoclass
                loaders = ['CIFAR100-test', 'CIFAR100-C-test-c4', 'CW-CIFAR100-test', 'BIM-CIFAR100-test', 'DF-CIFAR100-test', 'PGD-CIFAR100-test'], # these without the protoclass
                samples = idx,
                target_modules = target_layers,
                classes = Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta'),
                ticks = cp_ticks,
                protoclasses = protoclasses,
                scores = scores,
                verbose = verbose,
                )
