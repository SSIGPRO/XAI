import torch


def get_linear_conv2d_layers(net):
    return [
        name
        for name, module in net.named_modules()
        if isinstance(module, torch.nn.Linear)
        or isinstance(module, torch.nn.Conv2d)
    ]
