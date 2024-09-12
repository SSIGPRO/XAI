import torch
from datasets.cifar import Cifar
from models.vgg import VGG 
from activations.activations import Activations

from torchvision.models import vgg16, VGG16_Weights

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")
    
    # model parameters
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64

    ds = Cifar(dataset=dataset)

    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )
    
    pretrained = True
    model_dir = '/srv/newpenny/XAI/LM/models'
    model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
    f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'
    
    model = VGG(device=device)
    model.load_checkpoint(path=model_dir, name=model_name, verbose=True)
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    num_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model.set_model(model=nn)

    layers_dict = {'classifier': [0,3],
                  'features': [28]}
    direction = 'in'
    activations = Activations(model=model, dataset=ds)
    activations.compute_activations(
            layers_dict=layers_dict,
            direction=direction
            )