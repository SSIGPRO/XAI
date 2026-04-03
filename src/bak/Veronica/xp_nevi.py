import sys
sys.path.insert(0, '/home/lorenzo/repos/peepholelib')

# python stuff
from pathlib import Path as Path
from numpy.random import randint
from time import time
import pickle

from peepholelib.adv_atk.attacks_base import ftd

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
# from peepholelib.coreVectors.activations import binary_classification, multilabel_classification
from peepholelib.classifier.classifier_base import trim_corevectors
from peepholelib.classifier.tkmeans import KMeans as tKMeans 
from peepholelib.classifier.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.testing import trim_dataloaders


# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT
from torch.utils.data import DataLoader

class Mapping():
    def __init__(self, mapping=None, frame=None, col=None):

        if mapping is None:
            assert frame is not None and col is not None, 'not valid'
            mapping = self.to_mapping(frame, col)
        self.mapping = mapping
        pass

    def to_mapping(self, frame, col):
        _f = frame[col].to_list()
        mapping = dict(zip(frame[col].unique(), np.arange(len(frame[col].unique()))))
        self.col = col
        return  mapping

    def __call__(self, frame, col=None, modify_frame=False):
        if col is None:
            col = self.col

        new_col = np.array([self.mapping[d] for d in frame[col]])

        if modify_frame:
            frame[col] = new_col
            return frame
        else:
            return new_col
 

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    
    metadata = 'age'
    with open(file=f'/srv/newpenny/XAI/LC/Tesist*/VM/{metadata}_mapping_class.pkl', mode='rb') as f:
        mapping = pickle.load(f)
    print(mapping.mapping)
    filter_nan = False
    if 'non-valid' in mapping.mapping:
        filter_nan = True
        
    # ds_path = Path.cwd()/f'../data'
    ds_path = Path('/srv/newpenny/XAI/LC/Tesist*/VM/nevi_tensor')
    
    model_dir = Path('/srv/newpenny/XAI/VM/models')
    model_name = 'model_nevi.pth'

    verbose = True

    svds_path = Path.cwd()/f'../data/svds_nevi'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/f'../data/corevectors'
    cvs_name = 'corevectors_nevi'

    phs_path = Path.cwd()/f'../data/peepholes'
    phs_name = f'peepholes_nevi_{metadata}'

    #--------------------------------
    # Dataset 
    #--------------------------------
    bs = 64
    portion = ['train', 'val','test']
    ds = {}
    for ds_key in portion:
        ds[ds_key] = TensorDict.load_memmap(ds_path/f'{ds_key}_loader')
        
    #--------------------------------
    # Define the number of classes 
    #--------------------------------
    
    n_classes = len(mapping.mapping)

    #--------------------------------
    # Model 
    #--------------------------------

    model_custom = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    # Build a new classifier that reuses the pretrained layers and adds extra ones.
    new_classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(4096, 1024),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1024, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 1),
    )
    
    model_custom.classifier = new_classifier
    # file = Path('/srv/newpenny/XAI/VM/models/model_nevi.pth')
    
    model = ModelWrap(device=device)
    model.set_model(model=model_custom, path=model_dir, name=model_name, verbose=False)
            
    #--------------------------------
    # Model implementation 
    #--------------------------------
    
    target_layers = [
            # 'features.24', 
            # 'features.26', 
            # 'features.28', 
            'classifier.0',
            #'classifier.3',
            #'classifier.6',
            #'classifier.9',
            ]
    model.set_target_layers(target_layers=target_layers, verbose=True)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 

    # dry_img, _ = ds._train_ds.dataset[0]
    dry_img = ds['train']['image'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    
    print('target layers: ', model.get_target_layers()) 
    model.get_svds(path=svds_path, name=svds_name, verbose=verbose)
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)  
            
    #--------------------------------
    # Corevectors original
    #--------------------------------
    loaders = {key: DataLoader(value, batch_size=bs, collate_fn = lambda x: x, shuffle=False) for key, value in ds.items()} 
    loaders = trim_dataloaders(loaders, 0.05)

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )
    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
            loaders = loaders, 
            verbose = verbose,
            parser = ftd,
            key_list = list(ds['test'].keys())
            ) 
    
        cv.get_activations(
                batch_size = bs,
                loaders = loaders,
                verbose = verbose,
                # parser_pred = binary_classification
                )
    
        cv.get_coreVectors(
                batch_size = bs,
                reduct_matrices = model._svds,
                parser = parser_fn,
                verbose = verbose
                )
    
        cv.normalize_corevectors(
                wrt='train',
                verbose=verbose,
                to_file=Path(cvs_path)/(cvs_name+'.normalization.pt')
                )
        
        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['train']:
            print(data.keys())
            print(data['coreVectors']['classifier.0'].shape)
            print(data['coreVectors']['classifier.0'][34:56,:])
            print(f'Label {data['label']}')
            print(f'Pred  {data['pred']}')
            print(f'Result  {data['result']}')
            i += 1
            if i == 3: break

    parser_cv = trim_corevectors
    peep_layer = 'classifier.0'
    parser_kwargs = {'layer': peep_layer, 'peep_size':100, 'label': metadata} #
    cls_kwargs = {}#{'batch_size':256} 
    cls = tGMM(
            nl_classifier = 10,
            nl_model = n_classes,
            parser = parser_cv,
            parser_kwargs = parser_kwargs,
            cls_kwargs = cls_kwargs,
            device = device
            )

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name+'.'+peep_layer,
            classifier = cls,
            layer = peep_layer,
            device = device,
            filter_nan = filter_nan
            )

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        cv_dl = cv.get_dataloaders(
                batch_size = bs,
                verbose = True,
                )
    
        t0 = time()
        cls.fit(dataloader = cv_dl['train'], verbose=verbose)
        print('Fitting time = ', time()-t0)
        
        cls.compute_empirical_posteriors(verbose=verbose, filter_nan=filter_nan)
        
        ph.get_peepholes(
                loaders = cv_dl,
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )
        i = 0
        print('\nPrinting some peeps')
        ph_dl = ph.get_dataloaders(verbose=verbose)
        for data in ph_dl['test']:
            print('phs\n', data[peep_layer]['peepholes'].shape)
            print('phs\n', data[peep_layer]['peepholes'])
            print('max\n', data[peep_layer]['score_max'])
            print('ent\n', data[peep_layer]['score_entropy'])
            i += 1
            if i == 3: break


