import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# python stuff
from pathlib import Path as Path
import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from peepholelib.classifier.classifier_base import trim_corevectors
from peepholelib.classifier.tkmeans import KMeans as tKMeans 
from peepholelib.classifier.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16

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
    
    print('SIAMO NELLA peephole_wrap')
    
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
            mok, sok, mko, sko = ph.evaluate_dists(         # FUNZ VALUTAZIONE - ci da valutazione della metrica
                score_type = score_type,                    # poi in base ai val che trovi ne sceglie fi altri  -> prime iter sono aleatorie -> poi successive scelte secondo i risultati già ottenuti 
                coreVectors = cv_dl,
                bins = 20
                )

    else:
        print('Calcoliamo!! -> tGMM')
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
        data_path = Path(checkpoint_dir) / "tune_checkpoint.pkl"            # ste 3 righe non fanno nulla - si dovrebbero poter togliere
        with open(data_path, "wb") as fp:
            pickle.dump(ph_path, fp)
                                                               
        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report({                          # qui values passati al tuner
            'mok': mok['val'],
            'sok': sok['val'],
            'mko': mko['val'],
            'sko': sko['val'],
            # add AUC
            },
            checkpoint=checkpoint
        )

    return 

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    #device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    cuda_index = 0
    device = torch.device(f"cuda:{cuda_index}") if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    print(f"Use cuda paramenter: {use_cuda}")
    print(f"Cuda count device: {torch.cuda.device_count()}")
    # print(f"Nomi dei cuda device = {torch.cuda.get_device_name()}")


    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64 #512 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    xp_fold = 'xp_full_ds'
    
    # svds_name = 'svds' 
    # svds_path = Path.cwd()/'../data'
    svds_name = 'svd' 
    svds_path = f'/home/saravorabbi/Documents/{xp_fold}'
    
    # cvs_name = 'corevectors'
    # cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'
    cvs_path = Path(f'/home/saravorabbi/Documents/{xp_fold}/corevectors')
    
    # phs_name = 'peepholes'
    # phs_path = Path.cwd()/'../data/peepholes'
    # phs_path.mkdir(parents=True, exist_ok=True)
    phs_name = 'peepholes'
    phs_path = Path(f'/home/saravorabbi/Documents/{xp_fold}/peepholes/peep')
    phs_path.mkdir(parents=True, exist_ok=True)

    peep_layer = 'encoder.layers.encoder_layer_11.mlp.3'

    # corr_path = Path.cwd()/'temp_plots/correlations'
    # corr_path.mkdir(parents=True, exist_ok=True)
    corr_path = Path(f'/home/saravorabbi/Documents/{xp_fold}/temp_plots/correlations')
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

        config = {                                                              # configuration of the values -> taken randomly from the array (with tune.choice)
                #'peep_size': tune.choice([20*i for i in range(2, 16)]),         # peep_size -> dim of core vector
                'peep_size': tune.choice([20*i for i in range(2,38)]),     # range 40 - 760
                'n_classifier': tune.choice([20*i for i in range(2, 16)]),      # number of cluster 40-320
                #'score_type': tune.choice(['max', 'entropy']), 
                'score_type': tune.choice(['entropy']), 
                }

        if device == 'cpu':
            resources = {"cpu": 32}
        else:
            resources = {"cpu": 32, "gpu": 2}           # how many GPU we use - needs to be at least 1

        print(f"Abbiamo il device: {device} e le resources: {resources}")

        hyper_params_file = phs_path/f'hyperparams.{peep_layer}.pickle'     # file dove salviamo gli hyper params
        if hyper_params_file.exists():
            print("Already tunned parameters fount in %s. Skipping"%(hyper_params_file.as_posix()))
        else: 

            searcher = OptunaSearch(                            # metrics i want to use
                    metric = ['mok', 'sok', 'mko', 'sko'],          # our metrics are: mean and std - "mok" mean of ok elements, "sok" std of ok elements
                    mode = ['max', 'min', 'min', 'min']             # maximize the mean and minimize the std of ok elem
                    )
            algo = ConcurrencyLimiter(searcher, max_concurrent=4)
            scheduler = AsyncHyperBandScheduler(grace_period=5, max_t=100, metric="mok", mode="max") 
            tune_storage_path = Path(f'/home/saravorabbi/Documents/{xp_fold}/tuning')
            trainable = tune.with_resources(            # struttura interna dell'ottimizzatore
                    partial(
                        peephole_wrap,                  # funzione che fa le computazioni -> data una configurazione in config  -> compute metrics in searcher
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
                        num_samples = 30, #50,               # number of configuration the Tuner tries (use 1 or max 2 while deploying)
                        scheduler = scheduler,
                        ),
                    run_config = train.RunConfig(
                        storage_path = tune_storage_path
                        ),
                    param_space = config,
                    )
            print('PRE TUNER FIT')
            result = tuner.fit()
            print('POST TUNER FIT')

            results_df = result.get_dataframe()             # here are the results
            print('results: ', results_df)
            results_df.to_pickle(hyper_params_file)         # save file qui
    
    #---------------------------------
    # plot correlations between metrics and hyperparams 
    #---------------------------------
    print('\n------------------\nprinting\n------------------')
    hyperp_file = phs_path/f'hyperparams.{peep_layer}.pickle'       # vedi cosa c'è dentro (plottando) -> valori config di tutti i sample (50, nel mio caso 1) -> config e metriche calcolate
    rdf = pd.read_pickle(hyperp_file)
    
    print('RDF TIPO = ', type(rdf))
    print(rdf.head())
    
    
    
    metrics = np.vstack((rdf['mok'].values , rdf['sok'].values, rdf['mko'].values, rdf['sko'].values)).T
    m_names = ['mok', 'sok', 'mko', 'sko']
    configs = np.vstack((rdf['config/peep_size'].values, rdf['config/n_classifier'].values)).T              # è un pandas dataframe 
    c_names = ['peep_size', 'n_classifier']
    
    fig, axs = plt.subplots(2, 4, figsize=(4*4, 2*4))           # lo vedi col plot
    for m in range(metrics.shape[1]):
        for c in range(configs.shape[1]):
            ax = axs[c][m]
            df = pd.DataFrame({c_names[c]: configs[:,c], m_names[m]: metrics[:,m]})
            sb.scatterplot(data=df, ax=ax, x=c_names[c], y=m_names[m])
            ax.set_xlabel(c_names[c])
            ax.set_ylabel(m_names[m])
    plt.savefig((corr_path/peep_layer).as_posix()+'.png', dpi=300, bbox_inches='tight')
    plt.close()
