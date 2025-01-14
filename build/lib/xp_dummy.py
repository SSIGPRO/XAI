from pathlib import Path as Path
from datasets.dummy import Dummy
from models.dummy_model import DummyModel
from models.model_wrap import ModelWrap
from models.svd import get_svds

from models.viz import viz_singular_values, viz_compare, viz_compare_per_layer_type

#from activations.activations import Activations
#from peepholes.peepholes import Peepholes

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    # model parameters
    ds = Dummy()
    ds.load_data()

    #n_classes = len(train_loader.dataset.dataset.classes)

    #--------------------------------
    # Model 
    #--------------------------------

    # experiment with ViT_b_16
    model_dir = Path('/srv/newpenny/XAI/models/')
    model_name = Path('SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth')
    model_path = model_dir/model_name

    nn = torchvision.models.vit_b_16()
    
    # change the number of classes from 1000 to 100
    in_features = nn.heads.head.in_features
    n_classes = 100
    nn.heads.head = torch.nn.Linear(in_features, n_classes)

    # # ??
    # if not model_path.exists():
    #     torch.save({'state_dict': nn.state_dict()}, model_path)

    # print to check
    # for p in nn.parameters():
    #     print('nn parameters: ', p)


    wrap = ModelWrap()
    wrap.set_model(
        model = nn,
        path = model_dir,
        name = model_name
    )

    # print to check
    # for p in wrap._model.state_dict():
    #     print('nn parameters: ', p)



    # 1 - filter temporary dictionary
    st_list = list(nn.state_dict().keys())

    key_l = []

    for elem in st_list:
        if len(nn.state_dict()[elem].shape) == 2:
            key_l.append(elem)

    # remove .weight from the strings in the state_dict list
    key_clean = [s.replace(".weight", "") for s in key_l]
    key_clean = sorted(list(set(key_clean))) # unnecessary

    # 2 - set_target_layer
    wrap.set_target_layers(target_layers=key_clean)

    # 3 - call svd
    path_svd = Path('/home/saravorabbi/Documents/')
    name_svd = 'svd_prova'

    wrap.get_svds(path=path_svd, name=name_svd)

    # 5 - print img of curves of svd
    save_path = Path('/home/saravorabbi/Documents/viz_singular_values')
    viz_singular_values(wrap, save_path)

    save_path = Path('/home/saravorabbi/Documents/viz_sv_compare')
    viz_compare(wrap, save_path)

    save_path = Path('/home/saravorabbi/Documents/viz_sv_compare_per_layer_type')
    viz_compare_per_layer_type(wrap, save_path)
    
    





	#pretrained = True
	#model_dir = '/srv/newpenny/XAI/LM/models'
	#model_name = 'dummy_model.pth'

    # model_dir = Path('../data')
    # model_name = Path('banana.pt')
    # model_path = model_dir/model_name

    #nn = DummyModel(input_size=2, hidden_features=3, output_size=2)
    # if not model_path.exists():
    #     torch.save({'state_dict': nn.state_dict()}, model_path)

    # for p in nn.parameters():
    #     print('nn parameters: ', p)
    
    # dummy = ModelWrap()
    # dummy.set_model(
    #     model=nn,
    #     path=model_dir,
    #     name=model_name
    # )
    # for p in dummy._model.state_dict():
    #     print('nn parameters: ', p)


    # set target layer
    # target_layers_dict = {
    #     'nn3': {'banana': [1, 2]},
    #     'nn1':{
    #         'nn1': {'banana': [2]}, 
    #         'nn2':{'banana': [0, 1]}
    #         }
    #     }


    # # # # # # 1 - set_target_layer -> SONO I TARGET LAYER SU CUI COMPUTARE LA SVD
    # # # # # dummy.set_target_layers(layers_dict=False)
    # # # # # temp_dict = dummy._target_layers
    
    # # # # # # 2 - filter temporary dictionary
    # # # # # key_l = []
    
    # # # # # for elem in temp_dict:
    # # # # #     w_str = elem + ".weight"
    # # # # #     b_str = elem + ".bias"
        
    # # # # #     # create list of keys we want to delete later
    # # # # #     if len(dummy._model.state_dict()[w_str].shape) != 2:
    # # # # #         key_l.append(elem)

    # # # # # # delete all elem that has len != 2        
    # # # # # for _k in key_l:
    # # # # #     del temp_dict[_k]

    # # # # # print(temp_dict)


    # # # # # # 3 - take .weight .bias + concat

    # # # # # # 4 - call svd
    # # # # # # get_svds() -> look the function

    # # # # # # 5 - print img of curves of svd
    # # # # # # create this function






    # # set target layers
    # dummy.set_target_layers(target_layers=target_layers_dict)
    # for k in dummy._target_layers:
    #     layer = dummy._target_layers[k]
    #     print(k, ' ', layer.bias)

    # # add hooks
    # dummy.add_hooks(verbose=True)

    # # dry run
    # dry_input = ds._train_ds.dataset[0] #fai check per classe DataLoader -> .dataset[0]
    # print("LUNGHEZZA = ", type(dry_input))
    # print(type(dry_input[0]))
    # print(len(dry_input[0]))

    # #dry_input = dry_input.reshape((1,)+dry_img.shape)
    # #dummy.dry_run(x=dry_input)
    




