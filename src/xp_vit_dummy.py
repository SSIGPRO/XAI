import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# python
from pathlib import Path as Path

# peepholelib
# from peepholelib.datasets.dummy import Dummy
from peepholelib.datasets.cifar import Cifar

from peepholelib.models.dummy_model import DummyModel
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.svd import get_svds

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn

# torch
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR100

#from activations.activations import Activations
#from peepholes.peepholes import Peepholes

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1  # 5
    print('CUDA = ', cuda_index)
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    seed = 42
    verbose = True

    #--------------------------------
    # Directories definitions
    #--------------------------------
    cvs_name = 'corevectors'
    cvs_path = '/home/saravorabbi/Documents/corevectors'

    act_name = 'activations'

    phs_name = 'peepholes'
    phs_path = '/home/saravorabbi/Documents/peepholes'

    #--------------------------------
    # Dataset 
    #--------------------------------
    
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    dataset = 'CIFAR100'
    
    ds = Cifar(
        dataset=dataset,
        data_path=ds_path
    )
    ds.load_data(
        dataset=dataset,
        batch_size=64,
        data_kwargs = {'num_workers': 8, 'pin_memory': True},
        seed=seed
    )

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
    n_classes = len(ds._classes)
    in_features = nn.heads.head.in_features
    nn.heads.head = torch.nn.Linear(in_features, n_classes)

    # # ??
    # if not model_path.exists():
    #     torch.save({'state_dict': nn.state_dict()}, model_path)

    # print to check
    # for p in nn.parameters():
    #     print('nn parameters: ', p)

    wrap = ModelWrap(device=device)
    wrap.set_model(
        model = nn,
        path = model_dir,
        name = model_name
    )

    # print to check
    # for p in wrap._model.state_dict():
    #     print('nn parameters: ', p)

    # make list of target_layers
    st_list = list(nn.state_dict().keys())
    key_l = []
    for elem in st_list:
        if len(nn.state_dict()[elem].shape) == 2:
            key_l.append(elem)
    # remove .weight from the strings in the state_dict list
    target_layers = [s.replace(".weight", "") for s in key_l]
    target_layers = sorted(list(set(target_layers))) # unnecessary
    
    # TODO filter out self_attention.out_proj layers 
    
    isolated_target_layers = ['encoder.layers.encoder_layer_0.self_attention.out_proj'] 
    target_layer_mlp0 = ['encoder.layers.encoder_layer_0.mlp.0']
    
    #wrap.set_target_layers(target_layers=isolated_target_layers)
    wrap.set_target_layers(target_layers=target_layer_mlp0)

    print('Target Layers = ', wrap.get_target_layers())

    direction = {'save_input':True, 'save_output':False}
    wrap.add_hooks(**direction, verbose=verbose)

    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    wrap.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    path_svd = Path('/home/saravorabbi/Documents/')
    name_svd = 'svd_prova'
    wrap.get_svds(path=path_svd, name=name_svd)
    
    for k in wrap._svds.keys():
        for kk in wrap._svds[k].keys():
            print('svd shapes: ', k, kk, wrap._svds[k][kk].shape)


    #--------------------------------
    # CoreVectors 
    #--------------------------------

    ds_loaders = ds.get_dataset_loaders()

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = wrap,
        device = device
        )

    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
            loaders = ds_loaders,
            verbose = verbose
            )
        cv.get_activations(
            batch_size = 64,
            loaders = ds_loaders,
            verbose = verbose
            )
        cv.get_coreVectors(
            batch_size = 64,
            reduct_matrices = wrap._svds,
            parser = parser_fn,
            verbose = verbose
            )
        
        cv_dl = cv.get_dataloaders(verbose=verbose)

        #cv_act_loaders = 
        #act_load = cv.get_activations_loaders(verbose=verbose)
        
        
        #print('ACT LOAD = ', type(act_load))
        
        #print('TRAIN TYPE = ', type(act_load['train']))
        #print(act_load['train']['in_activations'])
        
        
        #layer = act_load['train'].dataset['in_activations']['encoder.layers.encoder_layer_0.mlp.0'].detach().cpu().numpy()
        #layer = prova.detach().cpu().numpy()
        #print(layer.shape)


    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                load_type = 'corevectors.activations',
                verbose = verbose
                ) 


        # act = cv._actds
        # print('ACT KEYS = ', act['train'].keys())
        # print('TIPO = ', type(act))

        # in_activations = act['train']['in_activations']
        # nested_keys = in_activations.keys()  # Check nested keys
        
        # print("IN ACTIVATION KEYS = ", nested_keys)
        
        # print(in_activations.keys())

        # layerr = in_activations['encoder.layers.encoder_layer_0.mlp.0']


        # if isinstance(layerr, torch.Tensor):
        #     numpy_data = layerr.detach().cpu().numpy()
        #     print('SONO DENTROOOO')
        #     print("Numpy array:", numpy_data.shape)


        #print some activations 
        #for p in act['train']['in_activations']





    print('ok :)')


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
    



