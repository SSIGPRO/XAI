import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision
from tensordict import TensorDict, PersistentTensorDict
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

########################################################################################################
#### In this .py we get the following info for the Test Set an we save in a Persistent Tensor Dict: ####
#### - from corevectors: image + true_label + pred_label + result                                   ####
#### - from model: output of the model                                                              ####
#### - from peepholes: conceptogram_matrix                                                          ####
########################################################################################################

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
    n_class = 100
    model.heads.head = torch.nn.Linear(in_features, n_class)

    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    model.to(device)
    model.eval()

    # corevectors
    cvs_path = Path(f'/home/saravorabbi/Documents/vit_1/corevectors')
    cvs_name = 'corevectors'

    cv = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )

    # peepholes
    phs_path = '/home/saravorabbi/Documents/vit_1/peepholes_ps_200_nc_150_full/'
    phs_name = 'peepholes'

    peep_dict = {}
    
    target_layers = [f'encoder.layers.encoder_layer_{i}.mlp.0' for i in range(12)]

    for peep_layer in target_layers:

        peephole = Peepholes(
            path = phs_path,
            name = phs_name + '.' + peep_layer,
            classifier = None,
            layer = peep_layer,
            device = device
            )

        peep_dict[peep_layer] = peephole    # add peephole to dictionary

    # -----------
    # tensordict
    # -----------
    n_samples = 10000   # vedi come trovare la dimensione del test set
    split_key = 'test'

    file_path = Path('/home/saravorabbi/Desktop/provona_concepto/' + split_key)
    print('filepath ', file_path)


    if file_path.exists():
        if verbose: print('Filepath exist - cancellalo')
        exit()
        # TODO impara a caricarlo
        # self._actds[ds_key] = PersistentTensorDict.from_h5(file_path, mode='r+')
        concept_ptd = PersistentTensorDict.from_h5(file_path, mode='r+')
    else:
        if verbose: print('Filepath does not exist')
        concept_ptd = PersistentTensorDict(filename=file_path, batch_size=[n_samples], mode='w')

    concept_ptd[split_key]['concept'] = MMT.empty(shape=torch.Size( (n_samples,) + (n_class,) + (len(target_layers),) ))
    concept_ptd[split_key]['outpt'] = MMT.empty(shape=torch.Size( (n_samples,) + (n_class,) + (len(target_layers),) ))

    #concept_td = concept_ptd[split_key]    # are we storing locally everything on the RAM with this
                                            # assegnamento? Does it save everything also in the PTD?
                                            # magari solo più leggebile

    with ExitStack() as stack:

        stack.enter_context(concept_ptd[split_key])

        concept_td = concept_ptd[split_key]


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
        
        # open all the 12 layer files for the peepholes
        ph_dl = {}
        for key in peep_dict:
            stack.enter_context(peep_dict[key])

            peep_dict[key].load_only(
                verbose = verbose,
                loaders = ['test']
            )
            ph_dl[key] = peep_dict[key].get_dataloaders(
                verbose = verbose
            )
        
        # TODO here call the function passing what you want to plot

        # iterate over cv and ph batches
        ph_dls = [ph_dl[f'encoder.layers.encoder_layer_{i}.mlp.0']['test'] for i in range(12)]  # fai un dizionario invece di una lista

        for batches in zip(cv_dl['test'], *(ph_dls)):
            cv_b = batches[0]
            layer_batches = batches[1:]

            label = -1
            pred = -1
            result = -1
            
            print('cv batch shape ', cv_b.shape[0])

            with torch.no_grad():
                output = model(cv_b['image'].to(device))
                #output = model(image.to(device))
            
            concept_ptd[split_key]['output'].append(output)

            # cycle on every element of the batch
            for elem in range(cv_b.shape[0]):   # We only need to cycle on cv dataloader size, since the batch sizes of the cv and of the ph are the same

                # print(layer_batches[0][0]['encoder.layers.encoder_layer_0.mlp.0']['peepholes'])
                # get info from cv
                image = cv_b[elem]['image']        
                label = cv_b[elem]['label']
                pred = cv_b[elem]['pred']
                result = cv_b[elem]['result']



                

                # get info from peepholes dataloader to generate the peephole matrix
                tensors = torch.tensor([])
            
                for i, batch in enumerate(layer_batches):                    
                    key_name = list(batch.keys())[0]    # key of the layer
                    tensor = batch[elem][key_name]['peepholes']
                    tensor = tensor.unsqueeze(1)
                    tensors = torch.cat((tensors, tensor), dim=1)
                
                matrix = np.column_stack(tensors.T)
                
                concept_ptd[split_key]['conceptogram'].append(matrix)

                

            break


    print('ciao')