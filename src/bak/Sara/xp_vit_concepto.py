import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision

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


if __name__ == "__main__":

    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 5  # torch.cuda.device_count() -1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    seed = 42
    verbose = True
    bs = 64

    # -----
    # path 
    # -----
    
    target_layers = [f'encoder.layers.encoder_layer_{i}.mlp.0' for i in range(12)]

    # model
    model_dir = '/srv/newpenny/XAI/models/'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    model = torchvision.models.vit_b_16()
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, 100)


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

    for peep_layer in target_layers:

        peephole = Peepholes(
            path = phs_path,
            name = phs_name + '.' + peep_layer,
            classifier = None,
            layer = peep_layer,
            device = device
            )
        
        # add peephole to dictionary
        peep_dict[peep_layer] = peephole

    
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
        

        

        # CHECKS
        a = True
        # now i have cv_dl and ph_dl -> these are the data loader of batch 64 that i can use to get data 
        for batch in cv_dl['test']:
            #print('cv ', batch.shape)
            if a == True:
                print('cv ', batch.shape)
                print(batch[1])
                a = False
        
        a = True
        for data in ph_dl['encoder.layers.encoder_layer_0.mlp.0']['test']:
            #print('ph ', data.shape)
            if a == True:
                print('ph ', data.shape)
                print(data[1])
                a = False
        
        exit()


        # ========================================================================
        # 13 dataloader?
        for cv_b, ph_l0, ph_l1, ph_l2, ph_l3, ph_l4, ph_l5, ph_l6, ph_l7, ph_l8, ph_l9, ph_l10, ph_l11 in zip(cv_dl['test'], ph_dl['encoder.layers.encoder_layer_0.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_1.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_2.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_3.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_4.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_5.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_6.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_7.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_8.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_9.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_10.mlp.0']['test'], ph_dl['encoder.layers.encoder_layer_11.mlp.0']['test']):
            
            label = -1
            pred = -1
            result = -1
            for i in 64:
                # get info from cv
                label = cv_b[i]['label']
                pred = cv_b[i]['pred']
                result = cv_b[i]['result']

                tensors = []

                # get tensor from 12 layers
                tensors.append(ph_l0[i]['encoder.layers.encoder_layer_0.mlp.0']['peepholes'])  # così prendo il peephole
                tensors.append(ph_l1[i]['encoder.layers.encoder_layer_1.mlp.0']['peepholes'])
                # etc

            # print('cv ', batch.shape)
            # print('ph ', data.shape)


        # ========================================================================

        # codice più intelligente forse
        ph_dls = [ph_dl[f'encoder.layers.encoder_layer_{i}.mlp.0']['test'] for i in range(12)]

        for batches in zip(cv_dl['test'], *(ph_dls)):
            cv_b = batches[0]
            layer_batches = batches[1:]

            # TODO here call the function passing what you want to plot
            
            label = -1
            pred = -1
            result = -1
            
            # cycle on every element of the current batch
            for i in range(cv_b.shape[0]):
                # get info from cv
                image = cv_b[i]['image']        # falla passare dentro al modello model.eval e salva output-> out = model(image)
                label = cv_b[i]['label']
                pred = cv_b[i]['pred']
                result = cv_b[i]['result']
                print('label = ', label)
                print('pred = ', pred)
                print('result = ', result)

                break

            # print('cv ', cv_b.shape)
            


            # PROVA 2
            for elem in range(64):

                tensors = torch.tensor([])
            
                for i, batch in enumerate(layer_batches):
                    #print(f'Layer {i}:', batch.shape)
                    #print(batch)
                    print('i ', i)
                    
                    key_name = list(batch.keys())[0]    # key of the layer
                    print('chiave = ', key_name)

                    print('shape = ', batch.shape)          #  torch.Size([64]) -> cat 64 alla volta e plotta immagini
                    print('type ', type(batch))
                    print('elem ', elem)
                    tensor = batch[elem][key_name]['peepholes']
                    
                    print('shape ', tensor.shape)
                    print('type ', type(tensor))
                    tensor = tensor.unsqueeze(1)

                    tensors = torch.cat((tensors, tensor), dim=1)
                
                #break
            

            # plot conceptogram img
            print('tensor ', tensors.shape)

            pred_value = int(pred.item())
            print('NUMEROPPPPP = ', pred_value)
            
            matrix = np.column_stack(tensors.T)
            
            dir_path = Path('/home/saravorabbi/Desktop/')
            img_name = Path('seconda.png')
            
            fig = plt.figure(figsize=(6, 10))
            ax = plt.gca()
            plt.imshow(matrix, aspect='auto', cmap='YlGnBu')
            plt.colorbar(label="Value Scale")
            plt.title(f'Elem: {elem} - True label: {int(label)}')
            plt.xlabel('ViT Layers')
            plt.ylabel('Classes')
            plt.xticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))

            true_value = int(label.flatten()[0].item())                        
            pred_value = int(pred.flatten()[0].item())
            plt.yticks([pred_value, true_value], [str(pred_value), str(true_value)])

            ax.yaxis.set_label_position("right")  # Move axis label
            ax.yaxis.tick_right()  # Move ticks

            plt.savefig(fname=dir_path/img_name, bbox_inches='tight')
            plt.close(fig)
        
            break

            
            # # PROVA 1
            # tensors = torch.tensor([])

            # for i, batch in enumerate(layer_batches):
            #     #print(f'Layer {i}:', batch.shape)
            #     #print(batch)
            #     print('i ', i)
                
            #     key_name = list(batch.keys())[0]    # key of the layer
            #     print('chiave = ', key_name)

            #     print('shape = ', batch.shape)          #  torch.Size([64]) -> cat 64 alla volta e plotta immagini
            #     print('type ', type(batch))

            #     tensor = batch[i][key_name]['peepholes']
            #     print('shape ', tensor.shape)
            #     print('type ', type(tensor))

            #     torch.cat(tensors, tensor, dim=1)
            #     #tensors.append(tensor)
            #     #break

            # print('tensor ', tensors.shape)




                
            # LABEL(49) OUT(68)  OUT_MAX 

            # import del modello
            # print ouput della rete

            # salvare matrix nel tensordict
            # ---------------------------------
            # funzione viz che carica il tensordict che visualizza quello che vuoi 








        # # work on the files
        # tensors = []

        # for key in peep_dict:
        #     tensor = peep_dict[key]._phs['test'][elem][key]['peepholes']
        #     tensors.append(tensor)
        
        # matrix = np.column_stack(tensors)
        
        # dir_path = Path('/home/saravorabbi/Desktop/')
        # img_name = Path('prova.png')

        # fig = plt.figure(figsize=(6, 10))
        
        # plt.imshow(matrix, aspect='auto', cmap='YlGnBu')
        # #plt.colorbar(label="Value Scale")
        # plt.title(f'Conceptogram elem {elem}')
        # plt.xlabel('ViT Layers')
        # plt.ylabel('Classes')
        # plt.xticks(np.arange(matrix.shape[1]), np.arange(matrix.shape[1]))
        # plt.savefig(fname=dir_path/img_name)
        # plt.close(fig)


    
    print('ciao')


# what i need to do
# 10.000 imgs

# -> trova lista label

# funzione che in base al flag sceglie:
# - resul==1 || result == 0
# - which label

# plot di sta roba