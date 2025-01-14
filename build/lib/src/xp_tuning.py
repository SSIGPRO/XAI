# python stuff
from pathlib import Path as Path
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np

# Our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from coreVectors.coreVectors import CoreVectors 
from coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from classifier.classifier_base import trim_corevectors
from classifier.tkmeans import KMeans as tKMeans 
from classifier.tgmm import GMM as tGMM 
from peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda

# Tuner
import tempfile
from functools import partial
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
import ray.cloudpickle as pickle

def peephole_wrap(config, **kwargs):
    peep_size = config['peep_size'] 
    n_cls = config['n_classifier']
    score_type = config['score_type']
    
    cv_dl = kwargs['corevec_dataloader']
    ph_path = kwargs['ph_path']
    ph_name = kwargs['ph_name']
    peep_layer = kwargs['peep_layer'] 

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = path(ph_path) / "tune_checkpoint.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]

    #--------------------------------
    # Peepholes
    #--------------------------------
    parser_cv = trim_corevectors
    n_classes = 100
    parser_kwargs = {'layer': peep_layer, 'peep_size':peep_size}
    cls_kwargs = {}#{'n_init':n_init, 'max_iter':max_iter} 
    ph_config_name = ph_name+f'.{peep_size}.{n_cls}.{score_type}'

    g = list(ph_path.glob(f'{ph_config_name}.*')) 
    if len(g) > 0:
        print('Already run this configuration, skipping peepholes computation')
        peepholes = Peepholes(
                path = ph_path,
                name = ph_config_name,
                classifier = None,
                layer = peep_layer,
                device = device
                )
        with peepholes as ph: 
            ph.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True
                    )
            
            # TODO: should save and load these instead of running the function again
            mok, sok, mko, sko = ph.evaluate_dists(
                score_type = score_type,
                coreVectors = cv_dl,
                bins = 20
                )

    else:
        cls = tGMM(
                nl_classifier = n_cls,
                nl_model = n_classes,
                parser = parser_cv,
                parser_kwargs = parser_kwargs,
                cls_kwargs = cls_kwargs,
                device = device
                )
                                                                         
        cls.fit(dataloader = cv_dl['train'], verbose=True)
        cls.compute_empirical_posteriors(verbose=True)
                                                                     
        peepholes = Peepholes(
                path = ph_path,
                name = ph_config_name,
                classifier = cls,
                layer = peep_layer,
                device = device
                )
        
        with peepholes as ph:
            ph.get_peepholes(
                loaders = cv_dl,
                verbose = True
                )
    
            ph.get_scores(
                    batch_size = 512, 
                    verbose=True
                    )

            mok, sok, mko, sko = ph.evaluate_dists(
                score_type = score_type,
                coreVectors = cv_dl,
                bins = 20
                )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "tune_checkpoint.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(ph_path, fp)
                                                               
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report({
            'mok': mok['val'],
            'sok': sok['val'],
            'mko': mko['val'],
            'sko': sko['val'],
            },
            checkpoint=checkpoint
        )

    return 

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_name = 'svds' 
    svds_path = Path.cwd()/'../data'
    
    cvs_name = 'corevectors'
    cvs_path = Path.cwd()/'../data/corevectors'
    
    phs_name = 'peepholes'
    phs_path = Path.cwd()/'../data/peepholes'
    phs_path.mkdir(parents=True, exist_ok=True)

    peep_layer = 'features.28'

    corr_path = Path.cwd()/'temp_plots/correlations'
    corr_path.mkdir(parents=True, exist_ok=True)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    with corevecs as cv: 
        # load corevds 
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        cv_dl = cv.get_dataloaders(batch_size=bs, verbose=True)
        i = 0

        #--------------------------------
        # Tunning 
        #--------------------------------

        config = {
                'peep_size': tune.choice([20*i for i in range(2, 16)]),
                'n_classifier': tune.choice([20*i for i in range(2, 16)]),
                #'score_type': tune.choice(['max', 'entropy']), 
                'score_type': tune.choice(['entropy']), 
                }

        if device == 'cpu':
            resources = {"cpu": 32}
        else:
            resources = {"cpu": 32, "gpu": 5}

        hyper_params_file = phs_path/f'hyperparams.{peep_layer}.pickle'
        if hyper_params_file.exists():
            print("Already tunned parameters fount in %s. Skipping"%(hyper_params_file.as_posix()))
        else: 

            searcher = OptunaSearch(
                    metric = ['mok', 'sok', 'mko', 'sko'],
                    mode = ['max', 'min', 'min', 'min']
                    )
            algo = ConcurrencyLimiter(searcher, max_concurrent=4)
            scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric="mok", mode="max") 
            tune_storage_path = Path.cwd()/'../data/tuning'
            trainable = tune.with_resources(
                    partial(
                        peephole_wrap,
                        device = device,
                        corevec_dataloader = cv_dl,
                        ph_path = phs_path,
                        ph_name = phs_name+'.'+peep_layer,
                        peep_layer = peep_layer 
                        ),
                    resources 
                    )

            tuner = tune.Tuner(
                    trainable,
                    tune_config = tune.TuneConfig(
                        search_alg = algo,
                        num_samples = 50, 
                        scheduler = scheduler,
                        ),
                    run_config = train.RunConfig(
                        storage_path = tune_storage_path
                        ),
                    param_space = config,
                    )
            result = tuner.fit()

            results_df = result.get_dataframe()
            print('results: ', results_df)
            results_df.to_pickle(hyper_params_file)
    
    #---------------------------------
    # plot correlations between metrics and hyperparams 
    #---------------------------------
    print('\n------------------\nprinting\n------------------')
    hyperp_file = phs_path/f'hyperparams.{peep_layer}.pickle'
    rdf = pd.read_pickle(hyperp_file)
    metrics = np.vstack((rdf['mok'].values , rdf['sok'].values, rdf['mko'].values, rdf['sko'].values)).T
    m_names = ['mok', 'sok', 'mko', 'sko']
    configs = np.vstack((rdf['config/peep_size'].values, rdf['config/n_classifier'].values)).T
    c_names = ['peep_size', 'n_classifier']
    
    fig, axs = plt.subplots(2, 4, figsize=(4*4, 2*4))
    for m in range(metrics.shape[1]):
        for c in range(configs.shape[1]):
            ax = axs[c][m]
            df = pd.DataFrame({c_names[c]: configs[:,c], m_names[m]: metrics[:,m]})
            sb.scatterplot(data=df, ax=ax, x=c_names[c], y=m_names[m])
            ax.set_xlabel(c_names[c])
            ax.set_ylabel(m_names[m])
    plt.savefig((corr_path/peep_layer).as_posix()+'.png', dpi=300, bbox_inches='tight')
    plt.close()
