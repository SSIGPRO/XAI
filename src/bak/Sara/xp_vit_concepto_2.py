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

                # plot of conceptogram
                matrix = np.column_stack(tensors.T)
                
                dir_path = Path('/home/saravorabbi/Desktop/')
                img_name = Path('seconda.png')
                
                fig = plt.figure(figsize=(6, 10))
                ax = plt.gca()
                plt.imshow(matrix, aspect='auto', cmap='YlGnBu')
                plt.colorbar(label="Value Scale")
                plt.title(f'Element: {elem} - True label: {int(label)} - Pred label: {int(pred)}')
                plt.xlabel('ViT Layers')
                plt.ylabel('Classes')
                plt.xticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))

                true_value = int(label.flatten()[0].item())                        
                pred_value = int(pred.flatten()[0].item())
                plt.yticks([pred_value, true_value], [str(pred_value), str(true_value)])

                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

                plt.savefig(fname=dir_path/img_name, bbox_inches='tight')
                plt.close(fig)

                
                break
                
            # print('tensor ', tensors.shape)
            # matrix = np.column_stack(tensors.T)


            # ===============================================
            # ======== CODICE PER PLOTTARTE IMMAGINE ========
            # ===============================================

            # pred_value = int(pred.item())
            # print('NUMEROPPPPP = ', pred_value)
            
            # matrix = np.column_stack(tensors.T)
            
            # dir_path = Path('/home/saravorabbi/Desktop/')
            # img_name = Path('seconda.png')
            
            # fig = plt.figure(figsize=(6, 10))
            # ax = plt.gca()
            # plt.imshow(matrix, aspect='auto', cmap='YlGnBu')
            # plt.colorbar(label="Value Scale")
            # plt.title(f'Elem: {elem} - True label: {int(label)}')
            # plt.xlabel('ViT Layers')
            # plt.ylabel('Classes')
            # plt.xticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))

            # true_value = int(label.flatten()[0].item())                        
            # pred_value = int(pred.flatten()[0].item())
            # plt.yticks([pred_value, true_value], [str(pred_value), str(true_value)])

            # ax.yaxis.set_label_position("right")  # Move axis label
            # ax.yaxis.tick_right()  # Move ticks

            # plt.savefig(fname=dir_path/img_name, bbox_inches='tight')
            # plt.close(fig)
        
            break


    
    print('ciao')