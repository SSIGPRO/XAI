import sys
from pathlib import Path as Path

from matplotlib import pyplot as plt
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from pathlib import Path as Path
from time import time
from functools import partial


# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.transforms import mobilenet_v2 as ds_transform 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import evaluate_dists 



# torch stuff
import torch
from cuda_selector import auto_cuda
import torchvision
import torch.nn as nn

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    #device = torch.device("cpu")
    #device = torch.device("cuda:4")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------

    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 128
    n_threads = 32

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

    svds_path = Path("/srv/newpenny/XAI/CN/data")
    svds_name = 'svds' 
    
    cvs_path = Path("/srv/newpenny/XAI/CN/data/corevectors")
    cvs_name = 'corevectors'

    drill_path = Path("/srv/newpenny/XAI/CN/data400cvdim300/drillers")
    drill_name = 'classifier'

    phs_path = Path("/srv/newpenny/XAI/CN/data400cvdim300/peepholes")
    phs_name = 'peepholes'

    verbose = True

    target_layers = [ 
        'features.1.conv.0.0', 'features.1.conv.1',
        'features.2.conv.0.0','features.2.conv.1.0', 'features.2.conv.2',
        'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
        'features.4.conv.0.0','features.4.conv.1.0', 'features.4.conv.2',
        'features.5.conv.0.0', 'features.5.conv.1.0','features.5.conv.2', 
        'features.6.conv.0.0', 'features.6.conv.1.0', 'features.6.conv.2', #B3
        'features.7.conv.0.0', 'features.7.conv.1.0', 'features.7.conv.2', #B3
        'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2', #B4
        'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2', #B4
        'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2', #B5
        'features.11.conv.0.0', 'features.11.conv.1.0','features.11.conv.2', #B5
        'features.12.conv.0.0', 'features.12.conv.1.0', 'features.12.conv.2', #B5
        'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2', #B5
        'features.14.conv.0.0', 'features.14.conv.1.0','features.14.conv.2', #B6
        'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2', #B6
        'features.16.conv.0.0', 'features.16.conv.1.0','features.16.conv.2', #B6
        'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2', #B7
        'features.18.0', 
        'classifier.1'
 ]
    
    cv_dim = 300
    n_cluster = 400

    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
        data_path = ds_path,
        dataset=dataset
        )

    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------

    nn = torchvision.models.mobilenet_v2()
    print(nn.state_dict().keys())
    n_classes = len(ds.get_classes()) 

    model = ModelWrap(
            model = nn,
            device = device
            )
    
    model.update_output(
            output_layer = 'classifier.1', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    model.set_target_modules(target_modules=target_layers, verbose=False)

        
    #--------------------------------
    # SVDs 
    #--------------------------------

    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            rank = 300,
            channel_wise = False,
            verbose = verbose
            )
     
    for k in model._svds.keys():
            for kk in model._svds[k].keys():
                print('svd shapes: ', k, kk, model._svds[k][kk].shape)
            s = model._svds[k]['s']
            if len(s.shape) == 1:
                plt.figure()
                plt.plot(s, '-')
                plt.xlabel('Rank')
                plt.ylabel('EigenVec')
            else:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
                for r in range(s.shape[0]):
                    plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
                ax.set_xlabel('Rank')
                ax.set_ylabel('Channel')
                ax.set_zlabel('EigenVec')
            plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
            plt.close()

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    dss = ds._dss
    #dss = random_subsampling(ds, 0.025)

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
    )

    reduction_fns = {

                'features.1.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.1.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.1.conv.0.0'],
                                device=device),
                'features.1.conv.1': partial(svd_Conv2D,
                                reduct_m=model._svds['features.1.conv.1']['Vh'], 
                                layer = model._target_modules['features.1.conv.1'],
                                device=device),
                'features.2.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.2.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.2.conv.0.0'],
                                device=device),
                'features.2.conv.1.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.2.conv.1.0']['Vh'], 
                                layer = model._target_modules['features.2.conv.1.0'],
                                device=device),
                'features.2.conv.2': partial(svd_Conv2D,
                                reduct_m=model._svds['features.2.conv.2']['Vh'], 
                                layer = model._target_modules['features.2.conv.2'],
                                device=device),
                
                'features.3.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.3.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.3.conv.0.0'],
                                device=device),
                'features.3.conv.1.0': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.3.conv.1.0']['Vh'], 
                                layer = model._target_modules['features.3.conv.1.0'],
                                device=device),
                'features.3.conv.2': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.3.conv.2']['Vh'], 
                                layer = model._target_modules['features.3.conv.2'],
                                device=device),
                'features.4.conv.0.0': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.4.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.4.conv.0.0'],
                                device=device),
                'features.4.conv.1.0': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.4.conv.1.0']['Vh'], 
                                layer = model._target_modules['features.4.conv.1.0'],
                                device=device),
                'features.4.conv.2': partial(svd_Conv2D,
                                reduct_m=model._svds['features.4.conv.2']['Vh'], 
                                layer = model._target_modules['features.4.conv.2'],
                                device=device),
                'features.5.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.5.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.5.conv.0.0'],
                                device=device),
                'features.5.conv.1.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.5.conv.1.0']['Vh'],
                                layer = model._target_modules['features.5.conv.1.0'],
                                device=device),
                'features.5.conv.2': partial(svd_Conv2D,
                                reduct_m=model._svds['features.5.conv.2']['Vh'], 
                                layer = model._target_modules['features.5.conv.2'],
                                device=device),
                'features.6.conv.0.0': partial(svd_Conv2D,        
                                   reduct_m=model._svds['features.6.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.6.conv.0.0'],
                                   device=device),
                'features.6.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.6.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.6.conv.1.0'],
                                   device=device),
                'features.6.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.6.conv.2']['Vh'], 
                                   layer = model._target_modules['features.6.conv.2'],
                                   device=device),
                'features.7.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.7.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.7.conv.0.0'],
                                   device=device),
                'features.7.conv.1.0': partial(svd_Conv2D,        
                                   reduct_m=model._svds['features.7.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.7.conv.1.0'],
                                   device=device), 
                'features.7.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.7.conv.2']['Vh'], 
                                   layer = model._target_modules['features.7.conv.2'],
                                   device=device),
                'features.8.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.8.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.8.conv.0.0'],
                                   device=device),

                'features.8.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.8.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.8.conv.1.0'],
                                   device=device), 
                 'features.8.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.8.conv.2']['Vh'], 
                                   layer = model._target_modules['features.8.conv.2'],
                                   device=device),     
                'features.9.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.9.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.9.conv.0.0'],
                                   device=device),              
                'features.9.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.9.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.9.conv.1.0'],
                                   device=device),
                'features.9.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.9.conv.2']['Vh'], 
                                   layer = model._target_modules['features.9.conv.2'],
                                   device=device),
                'features.10.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.10.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.10.conv.0.0'],
                                   device=device),
                'features.10.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.10.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.10.conv.1.0'],
                                   device=device),
                'features.10.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.10.conv.2']['Vh'], 
                                   layer = model._target_modules['features.10.conv.2'],
                                   device=device),
                'features.11.conv.0.0': partial(svd_Conv2D,     
                                          reduct_m=model._svds['features.11.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.11.conv.0.0'],
                                          device=device),
                'features.11.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.11.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.11.conv.1.0'],
                                   device=device),
                'features.11.conv.2': partial(svd_Conv2D,       
                                          reduct_m=model._svds['features.11.conv.2']['Vh'], 
                                          layer = model._target_modules['features.11.conv.2'],
                                          device=device),
                'features.12.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.12.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.12.conv.0.0'],
                                   device=device),
                'features.12.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.12.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.12.conv.1.0'],
                                   device=device),
                'features.12.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.12.conv.2']['Vh'], 
                                   layer = model._target_modules['features.12.conv.2'],
                                   device=device),   
                'features.13.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.13.conv.0.0'],
                                   device=device),                       
                'features.13.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.13.conv.1.0'],
                                   device=device),
                'features.13.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.2']['Vh'], 
                                   layer = model._target_modules['features.13.conv.2'],
                                   device=device),
                'features.14.conv.0.0': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.14.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.14.conv.0.0'],
                                          device=device),
                'features.14.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.14.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.14.conv.1.0'],
                                   device=device),
                'features.14.conv.2': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.14.conv.2']['Vh'], 
                                          layer = model._target_modules['features.14.conv.2'],
                                          device=device),
                'features.15.conv.0.0': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.15.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.15.conv.0.0'],
                                          device=device),
                'features.15.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.15.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.15.conv.1.0'],
                                   device=device),
                'features.15.conv.2': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.15.conv.2']['Vh'], 
                                          layer = model._target_modules['features.15.conv.2'],
                                          device=device),
                'features.16.conv.0.0': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.16.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.16.conv.0.0'],
                                          device=device),
                'features.16.conv.1.0': partial(svd_Conv2D,       
                                        reduct_m=model._svds['features.16.conv.1.0']['Vh'], 
                                        layer = model._target_modules['features.16.conv.1.0'],
                                        device=device), 
                'features.16.conv.2': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.16.conv.2']['Vh'], 
                                        layer = model._target_modules['features.16.conv.2'],
                                        device=device),
                'features.17.conv.0.0': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.17.conv.0.0']['Vh'], 
                                        layer = model._target_modules['features.17.conv.0.0'],
                                        device=device),
                'features.17.conv.1.0': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.17.conv.1.0']['Vh'], 
                                        layer = model._target_modules['features.17.conv.1.0'],
                                        device=device),
                'features.17.conv.2': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.17.conv.2']['Vh'], 
                                        layer = model._target_modules['features.17.conv.2'],
                                        device=device),
                'features.18.0': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.18.0']['Vh'], 
                                        layer = model._target_modules['features.18.0'],
                                        device=device),
                'classifier.1': partial(svd_Linear,
                                reduct_m=model._svds['classifier.1']['Vh'], 
                                device=device),  
            
    }


    with corevecs as cv: 
        # copy dataset to coreVect dataset

        cv.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )
        
        # cv.get_activations(
        #     batch_size = bs,
        #     n_threads = n_threads,
        #     save_input = True,
        #     save_output = False,
        #     verbose = verbose
        #     ) 
        
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )
        
        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'train',
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose=verbose
                    )
            
    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )


    cv_parsers = {

        'features.1.conv.0.0': partial(trim_corevectors,
                        module = 'features.1.conv.0.0',
                        cv_dim = cv_dim),
        'features.1.conv.1': partial(trim_corevectors,
                        module = 'features.1.conv.1',
                        cv_dim = cv_dim),
        'features.2.conv.0.0': partial(trim_corevectors,
                        module = 'features.2.conv.0.0',
                        cv_dim = cv_dim),
        'features.2.conv.1.0': partial(trim_corevectors,
                        module = 'features.2.conv.1.0',
                        cv_dim = cv_dim),
        'features.2.conv.2': partial(trim_corevectors,
                        module = 'features.2.conv.2',
                        cv_dim = cv_dim),
        'features.3.conv.0.0': partial(trim_corevectors,
                        module = 'features.3.conv.0.0',
                        cv_dim = cv_dim),
        'features.3.conv.1.0': partial(trim_corevectors,
                        module = 'features.3.conv.1.0',
                        cv_dim = cv_dim),
        'features.3.conv.2': partial(trim_corevectors,
                        module = 'features.3.conv.2',
                        cv_dim = cv_dim),
        'features.4.conv.0.0': partial(trim_corevectors,
                        module = 'features.4.conv.0.0',
                        cv_dim = cv_dim),
        'features.4.conv.1.0': partial(trim_corevectors,
                        module = 'features.4.conv.1.0',
                        cv_dim = cv_dim),
        'features.4.conv.2': partial(trim_corevectors,
                        module = 'features.4.conv.2',
                        cv_dim = cv_dim),
        'features.5.conv.0.0': partial(trim_corevectors,
                        module = 'features.5.conv.0.0',
                        cv_dim = cv_dim),
        'features.5.conv.1.0': partial(trim_corevectors,
                        module = 'features.5.conv.1.0',
                        cv_dim = cv_dim),
        'features.5.conv.2': partial(trim_corevectors,
                        module = 'features.5.conv.2',
                        cv_dim = cv_dim),
        'features.6.conv.0.0': partial(trim_corevectors,
                        module = 'features.6.conv.0.0',
                        cv_dim = cv_dim),
        'features.6.conv.1.0': partial(trim_corevectors,
                        module = 'features.6.conv.1.0',
                        cv_dim = cv_dim),
        'features.6.conv.2': partial(trim_corevectors,
                        module = 'features.6.conv.2',
                        cv_dim = cv_dim),
        'features.7.conv.0.0': partial(trim_corevectors,
                                module = 'features.7.conv.0.0',
                                cv_dim = cv_dim),
        'features.7.conv.1.0': partial(trim_corevectors,
                                module = 'features.7.conv.1.0',
                                cv_dim = cv_dim),
        'features.7.conv.2': partial(trim_corevectors,
                                module = 'features.7.conv.2',
                                cv_dim = cv_dim),
        'features.8.conv.0.0': partial(trim_corevectors,
                          module = 'features.8.conv.0.0',
                          cv_dim = cv_dim),
        'features.8.conv.1.0': partial(trim_corevectors,
                          module = 'features.8.conv.1.0',
                          cv_dim = cv_dim),
        'features.8.conv.2': partial(trim_corevectors,
                          module = 'features.8.conv.2',
                          cv_dim = cv_dim),
        'features.9.conv.0.0': partial(trim_corevectors,
                          module = 'features.9.conv.0.0',
                          cv_dim = cv_dim),
        'features.9.conv.1.0': partial(trim_corevectors,
                          module = 'features.9.conv.1.0',
                          cv_dim = cv_dim),
        'features.9.conv.2': partial(trim_corevectors,
                          module = 'features.9.conv.2',
                          cv_dim = cv_dim),
        'features.10.conv.0.0': partial(trim_corevectors,
                          module = 'features.10.conv.0.0',
                          cv_dim = cv_dim),
        'features.10.conv.1.0': partial(trim_corevectors,
                          module = 'features.10.conv.1.0',
                          cv_dim = cv_dim),
        'features.10.conv.2': partial(trim_corevectors,
                          module = 'features.10.conv.2',
                          cv_dim = cv_dim),
        'features.11.conv.0.0': partial(trim_corevectors,
                          module = 'features.11.conv.0.0',
                          cv_dim = cv_dim),
        'features.11.conv.1.0': partial(trim_corevectors,
                          module = 'features.11.conv.1.0',
                          cv_dim = cv_dim),
        'features.11.conv.2': partial(trim_corevectors,
                          module = 'features.11.conv.2',
                          cv_dim = cv_dim),
        'features.12.conv.0.0': partial(trim_corevectors,
                          module = 'features.12.conv.0.0',
                          cv_dim = cv_dim),
        'features.12.conv.1.0': partial(trim_corevectors,
                          module = 'features.12.conv.1.0',
                          cv_dim = cv_dim),
        'features.12.conv.2': partial(trim_corevectors,
                          module = 'features.12.conv.2',
                          cv_dim = cv_dim),
        'features.13.conv.0.0': partial(trim_corevectors,
                          module = 'features.13.conv.0.0',
                          cv_dim = cv_dim),
        'features.13.conv.1.0': partial(trim_corevectors,
                          module = 'features.13.conv.1.0',
                          cv_dim = cv_dim),
        'features.13.conv.2': partial(trim_corevectors,
                          module = 'features.13.conv.2',
                          cv_dim = cv_dim),
        'features.14.conv.0.0': partial(trim_corevectors,
                          module = 'features.14.conv.0.0',
                          cv_dim = cv_dim),
        'features.14.conv.1.0': partial(trim_corevectors,
                          module = 'features.14.conv.1.0',
                          cv_dim = cv_dim),   
        'features.14.conv.2': partial(trim_corevectors,
                          module = 'features.14.conv.2',
                          cv_dim = cv_dim),
        'features.15.conv.0.0': partial(trim_corevectors,
                          module = 'features.15.conv.0.0',
                          cv_dim = cv_dim),     
        'features.15.conv.1.0': partial(trim_corevectors,
                          module = 'features.15.conv.1.0',
                          cv_dim = cv_dim),
        'features.15.conv.2': partial(trim_corevectors,
                          module = 'features.15.conv.2',
                          cv_dim = cv_dim),
        'features.16.conv.0.0': partial(trim_corevectors,
                          module = 'features.16.conv.0.0',
                          cv_dim = cv_dim),
        'features.16.conv.1.0': partial(trim_corevectors,
                          module = 'features.16.conv.1.0',
                          cv_dim = cv_dim),
        'features.16.conv.2': partial(trim_corevectors,
                          module = 'features.16.conv.2',
                          cv_dim = cv_dim),
        'features.17.conv.0.0': partial(trim_corevectors,   
                          module = 'features.17.conv.0.0',
                          cv_dim = cv_dim),
        'features.17.conv.1.0': partial(trim_corevectors,
                          module = 'features.17.conv.1.0',
                          cv_dim = cv_dim),
        'features.17.conv.2': partial(trim_corevectors,
                          module = 'features.17.conv.2',
                          cv_dim = cv_dim),
        'features.18.0': partial(trim_corevectors,
                          module = 'features.18.0',
                          cv_dim = cv_dim),
        'classifier.1': partial(trim_corevectors,
                            module = 'classifier.1',
                            cv_dim = 100),
     }

    feature_sizes = {
        'features.1.conv.0.0': cv_dim,
        'features.1.conv.1': cv_dim,
        'features.2.conv.0.0': cv_dim,
        'features.2.conv.1.0': cv_dim,
        'features.2.conv.2': cv_dim,
        'features.3.conv.0.0': cv_dim,
        'features.3.conv.1.0': cv_dim,
        'features.3.conv.2': cv_dim,
        'features.4.conv.0.0': cv_dim,
        'features.4.conv.1.0': cv_dim,
        'features.4.conv.2': cv_dim,
        'features.5.conv.0.0': cv_dim,
        'features.5.conv.1.0': cv_dim,
        'features.5.conv.2': cv_dim,
        'features.6.conv.0.0': cv_dim,  
        'features.6.conv.1.0': cv_dim,
        'features.6.conv.2': cv_dim,
        'features.7.conv.0.0': cv_dim,
        'features.7.conv.1.0': cv_dim,
        'features.7.conv.2': cv_dim,
        'features.8.conv.0.0': cv_dim,
        'features.8.conv.1.0': cv_dim,
        'features.8.conv.2': cv_dim,
        'features.9.conv.0.0': cv_dim,
        'features.9.conv.1.0': cv_dim,
        'features.9.conv.2': cv_dim,
        'features.10.conv.0.0': cv_dim,
        'features.10.conv.1.0': cv_dim,
        'features.10.conv.2': cv_dim,   
        'features.11.conv.0.0': cv_dim,
        'features.11.conv.1.0': cv_dim,
        'features.11.conv.2': cv_dim,
        'features.12.conv.0.0': cv_dim,
        'features.12.conv.1.0': cv_dim,
        'features.12.conv.2': cv_dim,
        'features.13.conv.0.0': cv_dim,
        'features.13.conv.1.0': cv_dim,
        'features.13.conv.2': cv_dim,
        'features.14.conv.0.0': cv_dim,
        'features.14.conv.1.0': cv_dim,
        'features.14.conv.2': cv_dim,
        'features.15.conv.0.0': cv_dim,
        'features.15.conv.1.0': cv_dim,
        'features.15.conv.2': cv_dim,
        'features.16.conv.0.0': cv_dim,
        'features.16.conv.1.0': cv_dim,
        'features.16.conv.2': cv_dim,
        'features.17.conv.0.0': cv_dim,
        'features.17.conv.1.0': cv_dim,
        'features.17.conv.2': cv_dim,
        'features.18.0': cv_dim,
        'classifier.1': 100,
    }
    
    drillers = {}
    for peep_layer in target_layers:

        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parsers[peep_layer],
                device = device
                )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            device = device
            )
    
    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key} time = ', time()-t0)
                driller.fit(corevectors = cv._corevds['train'], verbose=verbose)

                driller.compute_empirical_posteriors(
                        dataset = cv._dss['train'],
                        corevectors = cv._corevds['train'],
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )
        print(ph._phs['train']['features.3.conv.1.0']['score_max'])
        evaluate_dists(
            peepholes = ph,
            score_type = 'max',
            dataset = cv._dss,
            bins = 20
            )

