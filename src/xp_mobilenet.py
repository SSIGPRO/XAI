import sys
#sys.path.insert(0, '/home/claranunesbarrancos/XAI/CN/peepholelib')
sys.path.insert(0, '/home/leandro/repos/peepholelib')

# python stuff
from pathlib import Path as Path
from peepholelib.datasets.cifar import Cifar
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from peepholelib.datasets.transforms import mobile_netv2 as mobile_net_transform

if __name__ == "__main__":
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds_path = '/srv/newpenny/dataset/CIFAR100'
    ds = Cifar(
            data_path = ds_path,
            dataset = dataset,
            transform = mobile_net_transform
            )

    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            original_transform = preprocess
            )
