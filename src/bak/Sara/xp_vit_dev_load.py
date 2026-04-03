import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

# python
import h5py
from pathlib import Path as Path
import numpy as np
from matplotlib import pyplot as plt
from contextlib import ExitStack
from itertools import islice

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.testing import trim_dataloaders


#### .py where we deploy the 9-th issue ####


if __name__ == "__main__":

    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 5  # torch.cuda.device_count() -1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    seed = 42
    verbose = True
    bs = 64
    
    
    # model
    model_dir = '/srv/newpenny/XAI/models/'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    model_path = model_dir + model_name
    
    model = torchvision.models.vit_b_16()
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, 100)

    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


    # corevectors
    cvs_path = Path(f'/home/saravorabbi/Documents/vit_1/corevectors')
    cvs_name = 'corevectors'

    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )

    # peepholes
    phs_path = Path(f'/home/saravorabbi/Desktop/9dev/peepholes_ps_200_nc_50_full')
    phs_name = 'peepholes'
    peep_size = 200
    n_cluster = 50

    peep_dict = {}
    
    # target_layers = [f'encoder.layers.encoder_layer_{i}.mlp.0' for i in range(12)]
    target_layers = ['encoder.layers.encoder_layer_10.mlp.0']#, 'encoder.layers.encoder_layer_11.mlp.0']
    #target_layers = ['encoder.layers.encoder_layer_11.mlp.0']

    # TODO for che genrea il dict of classifiers -> load the params of the models
    cl_dict = {}

    peephole = Peepholes(
        path = phs_path,
        name = f'{phs_name}.ps_{peep_size}.nc_{n_cluster}',
        classifiers = {}, # cl_dict
        layers = target_layers,
        device = device
        )

        # peep_dict[peep_layer] = peephole    # add peephole to dictionary

    
    with ExitStack() as stack:
        
        # open corevectors
        stack.enter_context(cv)

        cv.load_only(
            loaders = ['test'],
            verbose = verbose
        )

        cv_dl = cv.get_dataloaders(
            batch_size = bs,
            verbose = True,
        )
        
        stack.enter_context(peephole)

        peephole.load_only(
            verbose = verbose,
            loaders = ['test']
        )

        print('tutto ok')
        exit()








        # open all the 12 layer files for the peepholes
        ph_dl = {}
        #for key in peep_dict:
            #stack.enter_context(peep_dict[key])

        # stack.enter_context(peephole)

        # peephole.load_only(
        #     verbose = verbose,
        #     loaders = ['test']
        # )

        print('AFTER LOAD ONLY dei peepholes')
        print('peep _phs test: \n', peephole._phs['test'])

        
            
        # ph_dl[key] = peep_dict[key].get_dataloaders(
        #     verbose = verbose
        # )

        print('10: ', peephole._phs['test']['encoder.layers.encoder_layer_10.mlp.0'])


        print('tutto ok')
        exit()


















        # TODO here call the function passing what you want to plot
        # TODO BOH https://anonymous.4open.science/r/ATK-detection-783D/src/xp_discriminator_dataset.py

        # iterate over cv and ph batches
        ph_dls = [ph_dl[f'encoder.layers.encoder_layer_{i}.mlp.0']['test'] for i in range(12)]  # fai un dizionario invece di una lista

        for batches in zip(cv_dl['test'], *(ph_dls)):
            cv_b = batches[0]
            layer_batches = batches[1:]

            label = -1
            pred = -1
            result = -1
            
            print('cv batch shape ', cv_b.shape[0])

            # cycle on every element of the datset
            for elem in range(cv_b.shape[0]):   # We only need to cycle on cv dataloader size, since the batch sizes of the cv and of the ph are the same

                print(layer_batches[0][0]['encoder.layers.encoder_layer_0.mlp.0']['peepholes'])
                # get info from cv
                image = cv_b[elem]['image']        
                label = cv_b[elem]['label']
                pred = cv_b[elem]['pred']
                result = cv_b[elem]['result']
                # print('label = ', label)
                # print('pred = ', pred)
                # print('result = ', result)

                # TODO model.eval -> model(image)
                # get model output
                # out_model = model(image)

                # get info from peepholes dataloader to generate the peephole matrix
                tensors = torch.tensor([])
            
                for i, batch in enumerate(layer_batches):                    
                    key_name = list(batch.keys())[0]    # key of the layer
                    # print('chiave = ', key_name)
                    # print('shape = ', batch.shape)          #  torch.Size([64]) -> cat 64 alla volta e plotta immagini ???
                    # print('type ', type(batch))
                    # print('elem ', elem)
                    
                    tensor = batch[elem][key_name]['peepholes']
                    # print('shape ', tensor.shape)
                    # print('type ', type(tensor))

                    tensor = tensor.unsqueeze(1)

                    tensors = torch.cat((tensors, tensor), dim=1)

             
                break
                
            
            break


    
    print('ciao')