import io
from PIL import Image
from matplotlib import pyplot as plt

import torch
from pathlib import Path

if __name__ == '__main__':
    model_dir = Path('/srv/newpenny/XAI/models')
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

    ck = torch.load(model_dir/model_name, map_location='cpu')
    print('VGG16 CIFAR100--------------------')
    print(ck.keys())
    print('epoch: ', ck['epoch'])
    print('initial_lr: ', ck['initial_lr'])
    print('final_lr: ', ck['final_lr'])
    print('train_accuracy: ', ck['train_accuracy'])
    print('val_accuracy: ', ck['val_accuracy'])
    image = Image.open(io.BytesIO(ck['loss_plot']))
    plt.savefig('vgg16cifar100.png')
    
    model_name = 'vgg16_pretrained=True_dataset=CIFAR10-augmented_policy=CIFAR10_seed=29.pth'
    ck = torch.load(model_dir/model_name, map_location='cpu')
    print('VGG16 CIFAR10--------------------')
    print(ck.keys())
    print('epoch: ', ck['epoch'])
    print('lr: ', ck['lr'])
    print('batch_size: ', ck['batch_size'])
    print('train_accuracy: ', ck['train_accuracy'])
    print('val_accuracy: ', ck['val_accuracy'])
    print('train_loss: ', ck['train_loss'])
    print('val_loss: ', ck['val_loss'])


    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'
    ck = torch.load(model_dir/model_name)
    print('MobileNet CIFAR100--------------------')
    print(ck.keys())
    '''
    print('epoch: ', ck['epoch'])
    print('initial_lr: ', ck['initial_lr'])
    print('final_lr: ', ck['final_lr'])
    print('train_accuracy: ', ck['train_accuracy'])
    print('val_accuracy: ', ck['val_accuracy'])
    print(type(ck['loss_plot']))
    '''
