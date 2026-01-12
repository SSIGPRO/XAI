import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from functools import partial
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# torch stuff
import torch
from torchvision.models import vgg16 as Model
from cuda_selector import auto_cuda

# Model
from peepholelib.models.model_wrap import ModelWrap 
from torchvision.models import VGG16_Weights 
from peepholelib.datasets.functional.transforms import means, stds, vgg16_imagenet as transform
from peepholelib.datasets.CUB import CUB
from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

if __name__ == "__main__":

        #--------------------------------
        # Directories definitions
        #--------------------------------
        name_model = 'VGG16'
        name_dataset = 'CUB'
        normalizing_dataset = 'ImageNet'

        ds_path = Path.cwd()/f'../../data/parsed_datasets/{name_dataset}_{name_model}'
        #ds_path = f'/srv/newpenny/XAI/generated_data/TPAMI/parsed_datasets/{name_dataset}_{name_model}'

        # model parameters
        bs = 2**10 
        n_threads = 1
        n_classes = 1000

        svds_path = Path.cwd()/f'../../data/svds/{name_dataset}_{name_model}'
        #svds_path = Path(f'/srv/newpenny/XAI/generated_data/TPAMI/svds/{name_dataset}_{name_model}')
        svds_name = 'svds' 

        cvs_path = Path.cwd()/f'../../data/corevectors/{name_dataset}_{name_model}'
        # cvs_path = Path(f'/srv/newpenny/XAI/generated_data/TPAMI/corevectors/{name_dataset}_{name_model}')
        cvs_name = 'corevectors'

        verbose = True 

        save_input = True
        save_output = False

        # Peepholelib
        target_layers = [
                # 'features.7',
                # 'features.10',
                # 'features.12',
                # 'features.14',
                # 'features.17',
                # 'features.19',
                # 'features.21',
                # 'features.24',
                # 'features.26',
                # 'features.28',
                'classifier.0',
                'classifier.3',
                'classifier.6',
                ]

        target_layers = [f'model.{layer}' for layer in target_layers]

        svd_rank = 500 
        output_layer = 'classifier.6'

        loaders = [
                'CUB-train'
                'CUB-test'
                ]
        
        #--------------------------------
        # Dataset 
        #--------------------------------

        cub = CUB(path=ds_path)

        #--------------------------------
        # Model 
        #--------------------------------

        weights = VGG16_Weights.DEFAULT

        model = ModelWrap(
                model = Model(weights=weights),
                device = device
                )

        model.normalize_model(mean=means[normalizing_dataset], std= stds[normalizing_dataset])

        model.set_target_modules(
                target_modules = target_layers,
                verbose = verbose
                )

        #--------------------------------
        # SVDs 
        #--------------------------------
        svd_fns = {}
        for _layer in target_layers:
                if 'features' in _layer:
                        svd_fns[_layer] = partial(
                                conv2d_toeplitz_svd, 
                                rank = svd_rank, 
                                channel_wise = False,
                                device = device,
                                )
                elif 'classifier' in _layer:
                        svd_fns[_layer] = partial(
                                linear_svd,
                                rank = svd_rank,
                                device = device,
                                )
                
        # define a dimensionality reduction function for each layer
        reduction_fns = {}
        for _layer in target_layers:
                if 'features' in _layer:
                        reduction_fns[_layer] = partial(
                                conv2d_toeplitz_svd_projection, 
                                use_s = True,
                                device=device
                                )
                elif 'classifier' in _layer:
                        reduction_fns[_layer] = partial(
                                linear_svd_projection,
                                use_s = True,
                                device=device
                                )

        with cub as c:
                c.load_only(loaders=['train', 'test'])

                #--------------------------------
                # SVDs 
                #--------------------------------
                model.get_svds(
                        path = svds_path,
                        name = svds_name,
                        target_modules = target_layers,
                        sample_in = c._dss[f'{name_dataset}-train']['image'][0],
                        svd_fns = svd_fns,
                        verbose = verbose
                        )
                
                #--------------------------------
                # CoreVectors 
                #--------------------------------
                
                corevecs = CoreVectors(
                        path = cvs_path,
                        name = cvs_name,
                        model = model,
                        )
                
                with corevecs as cv: 
                        # add svd to reduction_fns
                        for _layer in reduction_fns:
                                if 'features' in _layer:
                                        reduction_fns[_layer].keywords['layer'] = model._target_modules[_layer]
                                reduction_fns[_layer].keywords['svd'] = model._svds[_layer] 
                        
                        # computing the corevectors
                        cv.get_coreVectors(
                                datasets = ds,
                                reduction_fns = reduction_fns,
                                save_input = save_input,
                                save_output = save_output,
                                batch_size = bs,
                                n_threads = n_threads,
                                verbose = verbose
                                )

                        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
                                cv.normalize_corevectors(
                                        wrt = f'{name_dataset}-train',
                                        to_file = cvs_path/(cvs_name+'.normalization.pt'),
                                        #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                                        batch_size = bs,
                                        n_threads = n_threads,
                                        verbose=verbose
                                        )

