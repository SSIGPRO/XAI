import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision

# python
from pathlib import Path as Path
from time import time

# peepholelib
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from peepholelib.utils.testing import trim_dataloaders
from peepholelib.classifier.classifier_base import trim_corevectors
from peepholelib.classifier.tgmm import GMM as tGMM
from peepholelib.peepholes.peepholes import Peepholes


#### .py where we deploy the 9-th issue ####

if __name__ == "__main__":
    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 4  # torch.cuda.device_count() -1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    seed = 42
    verbose = True
    bs = 512

    # ---------------
    # path
    # ---------------
    dataset = 'CIFAR100'
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    model_dir = '/srv/newpenny/XAI/models/'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    fold = 'vit_1'

    svd_path = f'/home/saravorabbi/Documents/{fold}'
    svd_name = 'svd'

    cvs_path = Path(f'/home/saravorabbi/Documents/{fold}/corevectors')
    cvs_name = 'corevectors'
    
    # ---------------
    # dataset
    # ---------------
    ds = Cifar(
        dataset=dataset,
        data_path=ds_path
    )

    ds.load_data(
        dataset=dataset,
        batch_size=512,
        data_kwargs = {'num_workers': 8, 'pin_memory': True},
        seed=seed
    )

    # ---------------
    # model wrap
    # ---------------    
    nn = torchvision.models.vit_b_16()
    in_features = nn.heads.head.in_features
    nn.heads.head = torch.nn.Linear(in_features, 100)

    wrap = ModelWrap(device=device)
    wrap.set_model(
        model = nn,
        path = model_dir,
        name = model_name
    )

    # target_layers = get_st_list(nn.state_dict().keys())
    target_layers = ['encoder.layers.encoder_layer_10.mlp.0', 'encoder.layers.encoder_layer_11.mlp.0']

    wrap.set_target_layers(target_layers=target_layers)
    
    direction = {'save_input':True, 'save_output':True}
    wrap.add_hooks(verbose=verbose)

    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    wrap.dry_run(x=dry_img)

    # ---------------
    # svd
    # ---------------
    wrap.get_svds(path=svd_path, name=svd_name)

    # ----------
    # peepholes computed only for mlp 0
    # ----------

    n_classes = 100
    parser_cv = trim_corevectors
    peep_size = 200
    n_cluster = 50


    corevecs = CoreVectors( 
        path = cvs_path,
        name = cvs_name,
    )

    phs_path = Path(f'/home/saravorabbi/Desktop/9dev/peepholes_ps_200_nc_{n_cluster}_full')
    phs_name = 'peepholes'
        
    # context manager
    with corevecs as cv:               # load only dei cv
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
        ) 
        cv_dl = cv.get_dataloaders(
            batch_size = bs,
            verbose = True,
        )
        
        # define the classifiers
        cls_dict = {}
        
        for peep_layer in target_layers:
            parser_kwargs = {'layer': peep_layer, 'peep_size':peep_size} # 'label_key': superclass -> vedi classifier_base.py -> def trim_corevectors(**kwargs):
            cls_kwargs = {}
            cls = tGMM(
                    nl_classifier = n_cluster,
                    nl_model = n_classes,
                    parser = parser_cv,
                    parser_kwargs = parser_kwargs,
                    cls_kwargs = cls_kwargs,
                    device = device
                    )
            
            cls_dict[peep_layer] = cls

        # define peephole class
        peepholes = Peepholes(
                    path = phs_path,
                    name = f'{phs_name}.ps_{peep_size}.nc_{n_cluster}',     # to change in peepholes.py | peepholes.ps_200.nc_50.test
                    classifiers = cls_dict,                                  # to change in peepholes.py
                    layers = target_layers,                                  # to change in peepholes.py
                    device = device
                    )
        
        
        # compute empirical posterior for every element in cls_dict
        for peep_layer in target_layers:
            t0 = time()
            cls_dict[peep_layer].fit(dataloader = cv_dl['train'], verbose=verbose)
            #cls.fit(dataloader = cv_dl['train'], verbose=verbose)
            print(f'Fitting time for layer {peep_layer} = ', time()-t0)

            cls_dict[peep_layer].compute_empirical_posteriors(verbose=verbose)

            # check what's inside
            print(cls_dict[peep_layer])



        with peepholes as ph:

            ph.get_peepholes(           # qua calcola i peepholes
                loaders = cv_dl,
                verbose = verbose
            )
            
            ph.save_classifiers(
                verbose = verbose,
            )

            # ph.get_classifier(
            #     device=device
            # )
            
            ph.get_scores(
                batch_size = bs,
                verbose=verbose
            )

            ph.evaluate_dists(
                score_type = 'max',
                coreVectors = cv_dl,
                bins = 20
            )

            print('peep _phs test: \n', ph._phs['test'])
        print('tutto ok')


 # here we compute an object Peephole for each layer
        # but we want an object Peephole with all the layers -> one file per ds (train, test, val)

        # -> We are going to have these 3 files
        # 'peepholes.train'
        # 'peepholes.test'
        # 'peepholes.val'
        # For each file, an object Peephole with 12 layers      <== this
        # Or for each file 12 object Peephole with one layer each
        #       encoder.layers.encoder_layer_0.mlp.0    -> peepholes 
        #                                               -> score
        #       encoder.layers.encoder_layer_1.mlp.0    -> peepholes 
        #                                               -> score
        #       etc ...
