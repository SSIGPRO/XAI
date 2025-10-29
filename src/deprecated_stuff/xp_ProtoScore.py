import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.cifar import Cifar

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import conceptogram_protoclass_score 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda
import pandas as pd

if __name__ == "__main__":
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    # name_model = 'ViT'
    name_model = 'vgg16'
    seed = 29
    bs = 64 
    n_threads = 32
    n_cluster = 150
    cv_dim = 100
    n_classes = 100
    
    cvs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/corevectors')
    # cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') #
    cvs_name = 'corevectors' # 'corevectors'

    drill_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/drillers')
    # drill_path = Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') #
    drill_name = 'classifier'

    phs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/peepholes')
    # phs_path = Path.cwd()/f'../data/{name_model}/peepholes_{cv_dim}' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') #
    phs_name = 'peepholes'
    
    verbose = True 
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )

    #--------------------------------
    # Peepholes
    #--------------------------------
    # target_layers = [ f'encoder.layers.encoder_layer_{i}.mlp.{j}'for i in range(12) for j in [0,3]]

    # target_layers.append('heads.head')
    
    target_layers = [
            #'features.0',
            #'features.2',
            #'features.5',
            #'features.7',
            'features.10',
            'features.12',
            'features.14',
            'features.17',
            'features.19',
            'features.21',
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]
    

    drillers = {}
    parser_cv = trim_corevectors

    # feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': cv_dim for i in range(12) for j in [0,3]}

    # feature_sizes['heads.head'] = cv_dim 

    features_cv_dim = 100
    classifier_cv_dim = 150

    feature_sizes = {}
    for _layer in target_layers:
        if 'features' in _layer:
                feature_sizes[_layer] =  features_cv_dim
        elif 'classifier' in _layer:
                feature_sizes[_layer] = classifier_cv_dim
    feature_sizes['classifier.6'] = n_classes
            

    drillers = {}
    for peep_layer in target_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = parser_cv,
                device = device
                )
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    peepholes = Peepholes(
            path = phs_path,
            name = f'{phs_name}', #.nc_{n_cluster}
            device = device
            )

    with corevecs as cv, peepholes as ph:
        ph.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                )
        cv.load_only(loaders = ['train', 'test', 'val'],
                     verbose = True
                     )
        '''
        dict_score = {}
        for i, target_layer in enumerate(target_layers):
            ret = conceptogram_protoclass_score(peepholes = ph,
                                            corevectors = cv,
                                            target_modules = [target_layer],
                                            proto_key = 'train',
                                            loaders = ['train', 'val', 'test'],
                                            plot = True,
                                            max_drop = 100,
                                            verbose = True,
                                            path = f'Bin/acc_vs_protoscore_{target_layer}.png'
                                            )
            
            if i==0: dict_score['model_conf'] = ret['SinglePercentage_conf']['test'].detach().cpu().numpy()

            dict_score[target_layer] = ret['SinglePercentage_score']['test'].detach().cpu().numpy()
        '''
        ret = conceptogram_protoclass_score(peepholes = ph,
                                            corevectors = cv,
                                            target_modules = target_layers,
                                            proto_key = 'train',
                                            loaders = ['train', 'val', 'test'],
                                            plot = True,
                                            max_drop = 100,
                                            verbose = True,
                                            path=f'mlVSsl_{name_model}.png')
        '''
        dict_score['multi_layer'] = ret['SinglePercentage_score']['test'].detach().cpu().numpy()

        df_percdrop = pd.DataFrame.from_dict(dict_score, orient='index',
                                             columns=['2%', '5%', '10%', '20%', '50%', '80%'])
        df_percdrop.to_csv(f'../data/{name_model}/percentageDrop_layer_by_layer_protoclass.csv')
        df_percdrop.to_latex(f'../data/{name_model}/percentageDrop_layer_by_layer_protoclass.tex')
        '''
