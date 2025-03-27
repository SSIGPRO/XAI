import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# General python stuff
from pathlib import Path as Path
from functools import partial
from time import time

# Our Stuff
from peepholelib.dummy.dummy_ds import DummyDS
from peepholelib.dummy.dummy_model import DummyModel
from peepholelib.dummy.functions import dummy_dim_reduction 

from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear

from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes


# Torch stuff
import torch
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    #--------------------------------
    # Definitions 
    #--------------------------------
    n = 100 # n samples
    ds = 10 # datasize
    bs = 25 # batch size
    verbose = True
    
    model_dir = Path.cwd()/'../data/dummy'
    model_name = 'model' 
    
    svds_path = Path.cwd()/'../data/dummy'
    svds_name = 'svds' 

    cvs_path = Path.cwd()/'../data/dummy'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../data/dummy'
    drill_name = 'classifier'
    
    phs_path = Path.cwd()/'../data/dummy'
    phs_name = 'peepholes'

    #--------------------------------
    # Dataset 
    #--------------------------------

    # model parameters
    dataset = DummyDS(n_samples=n, data_size=(ds,))
    dataset.load_data()
    dl = DataLoader(dataset._dss['d'], batch_size=bs)
    print('\n------- dummy dataset -------')
    for d in dl:
        print(d)

    #--------------------------------
    # Model 
    #--------------------------------
    model = ModelWrap(
            model = DummyModel(data_size=ds, hidden_features=3),
            path = model_dir,
            name = model_name,
            device = device
            )

    # print to check
    for p in model._model.state_dict():
         print('nn parameters: ', p)
    
    tm = ['nn1', 'nn2', 'nn3.banana.2']
    model.set_target_modules(target_modules=tm)
   
    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 

    dry_in, _ = dataset._dss['d'][0]
    dry_in = dry_in.reshape((1,)+dry_in.shape)
    model.dry_run(x=dry_in)
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    # We get the SVD just for one layer
    print('target modules: ', model.get_target_modules()) 
    model.get_svds(
            target_modules = ['nn3.banana.2'],
            path=svds_path,
            name=svds_name,
            verbose=verbose
            )

    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)

    #--------------------------------
    # CoreVectors 
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )

    shapes = {
            'nn1': 4,
            'nn2': 5,
            'nn3.banana.2': 6,
            }

    reduction_fns = {
            'nn1': partial(dummy_dim_reduction, size=shapes['nn1']),
            'nn2': partial(dummy_dim_reduction, size=shapes['nn2']),
            'nn3.banana.2': partial(dummy_dim_reduction, size=shapes['nn3.banana.2']),
            }

    with corevecs as cv: 
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                datasets = dataset._dss,
                verbose = verbose
                )        

        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                shapes = shapes,
                verbose = verbose
                )
        cv_dl = cv.get_dataloaders(verbose=verbose)

        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['d']:
            print('\nnn3.banana.2')
            print(data['nn3.banana.2'][:])
            i += 1
            if i == 1: break

    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = ds 
    n_cluster = 5 
    cv_dim = 3
    parser_cv = trim_corevectors
    peep_modules = ['nn1', 'nn2', 'nn3.banana.2']
    cls_kwargs = {}#{'batch_size': bs} 

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    drillers = {}
    for module in peep_modules:
        parser_kwargs = {'module': module, 'cv_dim':cv_dim}

        drillers[module] = tGMM(
                path = drill_path,
                name = drill_name+'.'+module,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = cv_dim,
                parser = parser_cv,
                parser_kwargs = parser_kwargs,
                device = device
                )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            driller = drillers,
            target_modules = peep_modules,
            device = device
            )

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['d'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            if (drill_path/(driller._suffix+'.empp.pt')).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key} time = ', time()-t0)
                driller.fit(corevectors = cv._corevds['d'], verbose=verbose)
                driller.compute_empirical_posteriors(
                        actds=cv._actds['d'],
                        corevds=cv._corevds['d'],
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['d'],
                verbose = True
                ) 

        ph.get_peepholes(
                corevectors = cv,
                batch_size = bs,
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )

        i = 0
        print('\nPrinting some peeps')
        ph_dl = ph.get_dataloaders(verbose=verbose)
        for data in ph_dl['d']:
            print('phs\n', data['nn3.banana.2']['peepholes'])
            print('max\n', data['nn3.banana.2']['score_max'])
            print('ent\n', data['nn3.banana.2']['score_entropy'])
            i += 1
            if i == 3: break
